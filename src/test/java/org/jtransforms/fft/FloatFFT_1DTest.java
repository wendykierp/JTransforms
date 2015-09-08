/* ***** BEGIN LICENSE BLOCK *****
 * JTransforms
 * Copyright (c) 2007 onward, Piotr Wendykier
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer. 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * ***** END LICENSE BLOCK ***** */
package org.jtransforms.fft;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;
import org.jtransforms.utils.CommonUtils;
import pl.edu.icm.jlargearrays.ConcurrencyUtils;
import org.jtransforms.utils.IOUtils;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import pl.edu.icm.jlargearrays.DoubleLargeArray;
import pl.edu.icm.jlargearrays.FloatLargeArray;
import pl.edu.icm.jlargearrays.LargeArray;
import static org.apache.commons.math3.util.FastMath.*;

/**
 * This is a series of JUnit tests for the {@link FloatFFT_1D}. First,
 * {@link FloatFFT_1D#complexForward(float[])} is tested by comparison with
 * reference data (FFTW). Then the other methods of this class are tested using
 * {@link FloatFFT_1D#complexForward(float[])} as a reference.
 *
 * @author S&eacute;bastien Brisard
 * @author Piotr Wendykier
 */
@RunWith(value = Parameterized.class)
public class FloatFFT_1DTest
{

    /**
     * Base message of all exceptions.
     */
    public static final String DEFAULT_MESSAGE = "%d-threaded FFT of size %d: ";

    /**
     * Name of binary files (input, untransformed data).
     */
    private final static String FFTW_INPUT_PATTERN = "fftw%d.in";

    /**
     * Name of binary files (output, transformed data).
     */
    private final static String FFTW_OUTPUT_PATTERN = "fftw%d.out";

    /**
     * The constant value of the seed of the random generator.
     */
    public static final int SEED = 20110602;

    private static final double EPS = pow(10, -4);

    private static final double EPS_UNSCALED = 0.5;

    @Parameters
    public static Collection<Object[]> getParameters()
    {
        final int[] size = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 32,
                            64, 100, 120, 128, 256, 310, 512, 1024, 1056, 2048, 8192,
                            10158, 16384, 32768, 65530, 65536, 131072};

        final ArrayList<Object[]> parameters = new ArrayList<Object[]>();
        for (int i = 0; i < size.length; i++) {
            parameters.add(new Object[]{size[i], 1, SEED});
            parameters.add(new Object[]{size[i], 2, SEED});
            parameters.add(new Object[]{size[i], 4, SEED});
        }
        return parameters;
    }

    /**
     * The FFT to be tested.
     */
    private final FloatFFT_1D fft;

    /**
     * The size of the FFT to be tested.
     */
    private final int n;

    /**
     * The number of threads used.
     */
    private final int numThreads;

    /**
     * For the generation of the data arrays.
     */
    private final Random random;

    /**
     * Creates a new instance of this class.
     *
     * @param n          the size of the FFT to be tested
     * @param numThreads the number of threads
     * @param seed       the seed of the random generator
     */
    public FloatFFT_1DTest(final int n, final int numThreads, final long seed)
    {
        this.n = n;
        this.fft = new FloatFFT_1D(n);
        LargeArray.setMaxSizeOf32bitArray(1);
        this.random = new Random(seed);
        CommonUtils.setThreadsBeginN_1D_FFT_2Threads(1024);
        CommonUtils.setThreadsBeginN_1D_FFT_4Threads(1024);
        ConcurrencyUtils.setNumberOfThreads(numThreads);
        this.numThreads = ConcurrencyUtils.getNumberOfThreads();
    }

    /**
     * Read the binary reference data files generated with FFTW. The structure
     * of these files is very simple: double values are written linearly (little
     * endian).
     *
     * @param name the file name
     * @param data the array to be updated with the data read (the size of this
     *             array gives the number of <code>double</code> to be retrieved
     */
    public void readData(final String name, final double[] data)
    {
        try {
            final File f = new File(getClass().getClassLoader()
                .getResource(name).getFile());
            final FileInputStream fin = new FileInputStream(f);
            final FileChannel fc = fin.getChannel();
            final ByteBuffer buffer = ByteBuffer.allocate(8 * data.length);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            fc.read(buffer);
            for (int i = 0; i < data.length; i++) {
                data[i] = buffer.getDouble(8 * i);
            }
        } catch (IOException e) {
            Assert.fail(e.getMessage());
        }
    }

    /**
     * Read the binary reference data files generated with FFTW. The structure
     * of these files is very simple: double values are written linearly (little
     * endian).
     *
     * @param name the file name
     * @param data the array to be updated with the data read (the size of this
     *             array gives the number of <code>double</code> to be retrieved
     */
    public void readData(final String name, final DoubleLargeArray data)
    {
        try {
            final File f = new File(getClass().getClassLoader()
                .getResource(name).getFile());
            final FileInputStream fin = new FileInputStream(f);
            final FileChannel fc = fin.getChannel();
            final ByteBuffer buffer = ByteBuffer.allocate(8 * (int) data.length());
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            fc.read(buffer);
            for (int i = 0; i < data.length(); i++) {
                data.setDouble(i, buffer.getDouble(8 * i));
            }
        } catch (IOException e) {
            Assert.fail(e.getMessage());
        }
    }

    /**
     * This is a test of {@link FloatFFT_1D#complexForward(float[])}. This
     * method is tested by computation of the FFT of some pre-generated data,
     * and comparison with results obtained with FFTW.
     */
    @Test
    public void testComplexForward()
    {
        final float[] actual = new float[2 * n];
        final double[] expected0 = new double[2 * n];
        readData(String.format(FFTW_INPUT_PATTERN, n), expected0);
        for (int index = 0; index < actual.length; index++) {
            actual[index] = (float) expected0[index];
        }
        readData(String.format(FFTW_OUTPUT_PATTERN, n), expected0);
        final float[] expected = new float[2 * n];
        for (int index = 0; index < expected.length; index++) {
            expected[index] = (float) expected0[index];
        }
        fft.complexForward(actual);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link FloatFFT_1D#complexForward(FloatLargeArray)}.
     * This method is tested by computation of the FFT of some pre-generated
     * data, and comparison with results obtained with FFTW.
     */
    @Test
    public void testComplexForwardLarge()
    {
        final FloatLargeArray actual = new FloatLargeArray(2 * n);
        final DoubleLargeArray expected0 = new DoubleLargeArray(2 * n);
        readData(String.format(FFTW_INPUT_PATTERN, n), expected0);
        for (int index = 0; index < actual.length(); index++) {
            actual.setFloat(index, expected0.getFloat(index));
        }
        readData(String.format(FFTW_OUTPUT_PATTERN, n), expected0);
        final FloatLargeArray expected = new FloatLargeArray(2 * n);
        for (int index = 0; index < expected.length(); index++) {
            expected.setFloat(index, expected0.getFloat(index));
        }
        fft.complexForward(actual);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link FloatFFT_1D#complexInverse(float[], boolean)},
     * with the second parameter set to <code>true</code>.
     */
    @Test
    public void testComplexInverseScaled()
    {
        final float[] actual = new float[2 * n];
        final float[] expected = new float[2 * n];
        for (int i = 0; i < 2 * n; i++) {
            actual[i] = 2.f * random.nextFloat() - 1.f;
            expected[i] = actual[i];
        }
        fft.complexForward(actual);
        fft.complexInverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of
     * {@link FloatFFT_1D#complexInverse(FloatLargeArray, boolean)}, with the
     * second parameter set to <code>true</code>.
     */
    @Test
    public void testComplexInverseScaledLarge()
    {
        final FloatLargeArray actual = new FloatLargeArray(2 * n);
        final FloatLargeArray expected = new FloatLargeArray(2 * n);
        for (int i = 0; i < 2 * n; i++) {
            actual.setFloat(i, 2.f * random.nextFloat() - 1.f);
            expected.setFloat(i, actual.getFloat(i));
        }
        fft.complexForward(actual);
        fft.complexInverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link FloatFFT_1D#complexInverse(float[], boolean)},
     * with the second parameter set to <code>false</code>.
     */
    @Test
    public void testComplexInverseUnscaled()
    {
        final float[] actual = new float[2 * n];
        final float[] expected = new float[2 * n];
        for (int i = 0; i < 2 * n; i++) {
            actual[i] = 2.f * random.nextFloat() - 1.f;
            expected[i] = actual[i];
        }
        fft.complexForward(actual);
        fft.complexInverse(actual, false);
        final float s = 1.f / (float) n;
        for (int i = 0; i < actual.length; i++) {
            actual[i] = s * actual[i];
        }
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS_UNSCALED);
    }

    /**
     * This is a test of
     * {@link FloatFFT_1D#complexInverse(FloatLargeArray, boolean)}, with the
     * second parameter set to <code>false</code>.
     */
    @Test
    public void testComplexInverseUnscaledLarge()
    {
        final FloatLargeArray actual = new FloatLargeArray(2 * n);
        final FloatLargeArray expected = new FloatLargeArray(2 * n);
        for (int i = 0; i < 2 * n; i++) {
            actual.setFloat(i, 2.f * random.nextFloat() - 1.f);
            expected.setFloat(i, actual.getFloat(i));
        }
        fft.complexForward(actual);
        fft.complexInverse(actual, false);
        final float s = 1.f / (float) n;
        for (int i = 0; i < actual.length(); i++) {
            actual.setFloat(i, s * actual.getFloat(i));
        }
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS_UNSCALED);
    }

    /**
     * This is a test of {@link FloatFFT_1D#realForward(float[])}.
     */
    @Test
    public void testRealForward()
    {
        final float[] actual = new float[2 * n];
        final float[] expected = new float[2 * n];
        for (int i = 0; i < n; i++) {
            actual[i] = 2.f * random.nextFloat() - 1.f;
            expected[2 * i] = actual[i];
            expected[2 * i + 1] = 0.f;
        }
        fft.complexForward(expected);
        fft.realForward(actual);
        if (!CommonUtils.isPowerOf2(n)) {
            int m;
            if (n % 2 == 0) {
                m = n / 2;
            } else {
                m = (n + 1) / 2;
            }
            actual[n] = actual[1];
            actual[1] = 0;
            for (int k = 1; k < m; k++) {
                int idx1 = 2 * n - 2 * k;
                int idx2 = 2 * k;
                actual[idx1] = actual[idx2];
                actual[idx1 + 1] = -actual[idx2 + 1];
            }
        } else {
            for (int k = 1; k < n / 2; k++) {
                int idx1 = 2 * n - 2 * k;
                int idx2 = 2 * k;
                actual[idx1] = actual[idx2];
                actual[idx1 + 1] = -actual[idx2 + 1];
            }
            actual[n] = actual[1];
            actual[1] = 0;
        }

        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link FloatFFT_1D#realForward(FloatLargeArray)}.
     */
    @Test
    public void testRealForwardLarge()
    {
        final FloatLargeArray actual = new FloatLargeArray(2 * n);
        final FloatLargeArray expected = new FloatLargeArray(2 * n);
        for (int i = 0; i < n; i++) {
            actual.setFloat(i, 2.f * random.nextFloat() - 1.f);
            expected.setFloat(2 * i, actual.getFloat(i));
            expected.setFloat(2 * i + 1, 0.0f);
        }
        fft.complexForward(expected);
        fft.realForward(actual);
        if (!CommonUtils.isPowerOf2(n)) {
            int m;
            if (n % 2 == 0) {
                m = n / 2;
            } else {
                m = (n + 1) / 2;
            }
            actual.setFloat(n, actual.getFloat(1));
            actual.setFloat(1, 0);
            for (int k = 1; k < m; k++) {
                int idx1 = 2 * n - 2 * k;
                int idx2 = 2 * k;
                actual.setFloat(idx1, actual.getFloat(idx2));
                actual.setFloat(idx1 + 1, -actual.getFloat(idx2 + 1));
            }
        } else {
            for (int k = 1; k < n / 2; k++) {
                int idx1 = 2 * n - 2 * k;
                int idx2 = 2 * k;
                actual.setFloat(idx1, actual.getFloat(idx2));
                actual.setFloat(idx1 + 1, -actual.getFloat(idx2 + 1));
            }
            actual.setFloat(n, actual.getFloat(1));
            actual.setFloat(1, 0);
        }

        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link FloatFFT_1D#realForward(float[])}.
     */
    @Test
    public void testRealForwardFull()
    {
        final float[] actual = new float[2 * n];
        final float[] expected = new float[2 * n];
        for (int i = 0; i < n; i++) {
            actual[i] = 2.f * random.nextFloat() - 1.f;
            expected[2 * i] = actual[i];
            expected[2 * i + 1] = 0.f;
        }
        fft.complexForward(expected);
        fft.realForwardFull(actual);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link FloatFFT_1D#realForward(FloatLargeArray)}.
     */
    @Test
    public void testRealForwardFullLarge()
    {
        final FloatLargeArray actual = new FloatLargeArray(2 * n);
        final FloatLargeArray expected = new FloatLargeArray(2 * n);
        for (int i = 0; i < n; i++) {
            actual.setFloat(i, 2.f * random.nextFloat() - 1.f);
            expected.setFloat(2 * i, actual.getFloat(i));
            expected.setFloat(2 * i + 1, 0.0f);
        }
        fft.complexForward(expected);
        fft.realForwardFull(actual);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link FloatFFT_1D#realInverseFull(float[], boolean)} ,
     * with the second parameter set to <code>true</code>.
     */
    @Test
    public void testRealInverseFullScaled()
    {
        final float[] actual = new float[2 * n];
        final float[] expected = new float[2 * n];
        for (int i = 0; i < n; i++) {
            actual[i] = 2.f * random.nextFloat() - 1.f;
            expected[2 * i] = actual[i];
            expected[2 * i + 1] = 0.f;
        }
        fft.realInverseFull(actual, true);
        fft.complexInverse(expected, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of
     * {@link FloatFFT_1D#realInverseFull(FloatLargeArray, boolean)} , with the
     * second parameter set to <code>true</code>.
     */
    @Test
    public void testRealInverseFullScaledLarge()
    {
        final FloatLargeArray actual = new FloatLargeArray(2 * n);
        final FloatLargeArray expected = new FloatLargeArray(2 * n);
        for (int i = 0; i < n; i++) {
            actual.setFloat(i, 2.f * random.nextFloat() - 1.f);
            expected.setFloat(2 * i, actual.getFloat(i));
            expected.setFloat(2 * i + 1, 0.0f);
        }
        fft.realInverseFull(actual, true);
        fft.complexInverse(expected, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link FloatFFT_1D#realInverseFull(float[], boolean)} ,
     * with the second parameter set to <code>false</code>.
     */
    @Test
    public void testRealInverseFullUnscaled()
    {
        final float[] actual = new float[2 * n];
        final float[] expected = new float[2 * n];
        for (int i = 0; i < n; i++) {
            actual[i] = 2.f * random.nextFloat() - 1.f;
            expected[2 * i] = actual[i];
            expected[2 * i + 1] = 0.f;
        }
        fft.realInverseFull(actual, false);
        fft.complexInverse(expected, false);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS_UNSCALED);
    }

    /**
     * This is a test of
     * {@link FloatFFT_1D#realInverseFull(FloatLargeArray, boolean)} , with the
     * second parameter set to <code>false</code>.
     */
    @Test
    public void testRealInverseFullUnscaledLarge()
    {
        final FloatLargeArray actual = new FloatLargeArray(2 * n);
        final FloatLargeArray expected = new FloatLargeArray(2 * n);
        for (int i = 0; i < n; i++) {
            actual.setFloat(i, 2.f * random.nextFloat() - 1.f);
            expected.setFloat(2 * i, actual.getFloat(i));
            expected.setFloat(2 * i + 1, 0.0f);
        }
        fft.realInverseFull(actual, false);
        fft.complexInverse(expected, false);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS_UNSCALED);
    }

    /**
     * This is a test of {@link FloatFFT_1D#realInverse(float[], boolean)}, with
     * the second parameter set to <code>true</code>.
     */
    @Test
    public void testRealInverseScaled()
    {
        final float[] actual = new float[n];
        final float[] expected = new float[n];
        for (int i = 0; i < n; i++) {
            actual[i] = 2.f * random.nextFloat() - 1.f;
            expected[i] = actual[i];
        }
        fft.realForward(actual);
        fft.realInverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of
     * {@link FloatFFT_1D#realInverse(FloatLargeArray, boolean)}, with the
     * second parameter set to <code>true</code>.
     */
    @Test
    public void testRealInverseScaledLarge()
    {
        final FloatLargeArray actual = new FloatLargeArray(n);
        final FloatLargeArray expected = new FloatLargeArray(n);
        for (int i = 0; i < n; i++) {
            actual.setFloat(i, 2.f * random.nextFloat() - 1.f);
            expected.setFloat(i, actual.getFloat(i));
        }
        fft.realForward(actual);
        fft.realInverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link FloatFFT_1D#realInverse(float[], boolean)}, with
     * the second parameter set to <code>false</code>.
     */
    @Test
    public void testRealInverseUnscaled()
    {
        final float[] actual = new float[n];
        final float[] expected = new float[n];
        for (int i = 0; i < n; i++) {
            actual[i] = 2.f * random.nextFloat() - 1.f;
            expected[i] = actual[i];
        }
        fft.realForward(actual);
        fft.realInverse(actual, false);
        float s;
        if (CommonUtils.isPowerOf2(n) && n > 1) {
            s = 2.f / (float) n;
        } else {
            s = 1.f / (float) n;
        }
        for (int i = 0; i < actual.length; i++) {
            actual[i] = s * actual[i];
        }
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS_UNSCALED);
    }

    /**
     * This is a test of
     * {@link FloatFFT_1D#realInverse(FloatLargeArray, boolean)}, with the
     * second parameter set to <code>false</code>.
     */
    @Test
    public void testRealInverseUnscaledLarge()
    {
        final FloatLargeArray actual = new FloatLargeArray(n);
        final FloatLargeArray expected = new FloatLargeArray(n);
        for (int i = 0; i < n; i++) {
            actual.setFloat(i, 2.f * random.nextFloat() - 1.f);
            expected.setFloat(i, actual.getFloat(i));
        }
        fft.realForward(actual);
        fft.realInverse(actual, false);
        float s;
        if (CommonUtils.isPowerOf2(n) && n > 1) {
            s = 2.f / (float) n;
        } else {
            s = 1.f / (float) n;
        }
        for (int i = 0; i < actual.length(); i++) {
            actual.setFloat(i, s * actual.getFloat(i));
        }
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS_UNSCALED);
    }
}
