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

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import org.jtransforms.utils.CommonUtils;
import pl.edu.icm.jlargearrays.ConcurrencyUtils;
import org.jtransforms.utils.IOUtils;
import pl.edu.icm.jlargearrays.DoubleLargeArray;
import pl.edu.icm.jlargearrays.LargeArray;
import static org.apache.commons.math3.util.FastMath.*;

/**
 * This is a series of JUnit tests for the {@link DoubleFFT_1D}. First,
 * {@link DoubleFFT_1D#complexForward(double[])} is tested by comparison with
 * reference data (FFTW). Then the other methods of this class are tested using
 * {@link DoubleFFT_1D#complexForward(double[])} as a reference.
 *
 * @author S&eacute;bastien Brisard
 * @author Piotr Wendykier
 */
@RunWith(value = Parameterized.class)
public class DoubleFFT_1DTest
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

    private static final double EPS = pow(10, -12);

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
    private final DoubleFFT_1D fft;

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
     * @param n
     *                   the size of the FFT to be tested
     * @param numThreads
     *                   the number of threads
     * @param seed
     *                   the seed of the random generator
     */
    public DoubleFFT_1DTest(final int n, final int numThreads, final long seed)
    {
        this.n = n;
        LargeArray.setMaxSizeOf32bitArray(1);
        this.fft = new DoubleFFT_1D(n);
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
     * @param name
     *             the file name
     * @param data
     *             the array to be updated with the data read (the size of this
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
     * @param name
     *             the file name
     * @param data
     *             the array to be updated with the data read (the size of this
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
     * This is a test of {@link DoubleFFT_1D#complexForward(double[])}. This
     * method is tested by computation of the FFT of some pre-generated data,
     * and comparison with results obtained with FFTW.
     */
    @Test
    public void testComplexForward()
    {
        final double[] actual = new double[2 * n];
        final double[] expected = new double[2 * n];
        readData(String.format(FFTW_INPUT_PATTERN, n), actual);
        readData(String.format(FFTW_OUTPUT_PATTERN, n),
                 expected);
        fft.complexForward(actual);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link DoubleFFT_1D#complexForward(DoubleLargeArray)}. This
     * method is tested by computation of the FFT of some pre-generated data,
     * and comparison with results obtained with FFTW.
     */
    @Test
    public void testComplexForwardLarge()
    {
        final DoubleLargeArray actual = new DoubleLargeArray(2 * n);
        final DoubleLargeArray expected = new DoubleLargeArray(2 * n);
        readData(String.format(FFTW_INPUT_PATTERN, n), actual);
        readData(String.format(FFTW_OUTPUT_PATTERN, n), expected);
        fft.complexForward(actual);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link DoubleFFT_1D#complexInverse(double[], boolean)},
     * with the second parameter set to <code>true</code>.
     */
    @Test
    public void testComplexInverseScaled()
    {
        final double[] actual = new double[2 * n];
        final double[] expected = new double[2 * n];
        for (int i = 0; i < 2 * n; i++) {
            actual[i] = 2. * random.nextDouble() - 1.;
            expected[i] = actual[i];
        }
        fft.complexForward(actual);
        fft.complexInverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link DoubleFFT_1D#complexInverse(DoubleLargeArray, boolean)},
     * with the second parameter set to <code>true</code>.
     */
    @Test
    public void testComplexInverseScaledLarge()
    {
        final DoubleLargeArray actual = new DoubleLargeArray(2 * n);
        final DoubleLargeArray expected = new DoubleLargeArray(2 * n);
        for (int i = 0; i < 2 * n; i++) {
            actual.setDouble(i, 2. * random.nextDouble() - 1.);
            expected.setDouble(i, actual.getDouble(i));
        }
        fft.complexForward(actual);
        fft.complexInverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link DoubleFFT_1D#complexInverse(double[], boolean)},
     * with the second parameter set to <code>false</code>.
     */
    @Test
    public void testComplexInverseUnscaled()
    {
        final double[] actual = new double[2 * n];
        final double[] expected = new double[2 * n];
        for (int i = 0; i < 2 * n; i++) {
            actual[i] = 2. * random.nextDouble() - 1.;
            expected[i] = actual[i];
        }
        fft.complexForward(actual);
        fft.complexInverse(actual, false);
        final double s = 1. / (double) n;
        for (int i = 0; i < actual.length; i++) {
            actual[i] = s * actual[i];
        }
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link DoubleFFT_1D#complexInverse(DoubleLargeArray, boolean)},
     * with the second parameter set to <code>false</code>.
     */
    @Test
    public void testComplexInverseUnscaledLarge()
    {
        final DoubleLargeArray actual = new DoubleLargeArray(2 * n);
        final DoubleLargeArray expected = new DoubleLargeArray(2 * n);
        for (int i = 0; i < 2 * n; i++) {
            actual.setDouble(i, 2. * random.nextDouble() - 1.);
            expected.setDouble(i, actual.getDouble(i));
        }
        fft.complexForward(actual);
        fft.complexInverse(actual, false);
        final double s = 1. / (double) n;
        for (int i = 0; i < actual.length(); i++) {
            actual.setDouble(i, s * actual.getDouble(i));
        }
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link DoubleFFT_1D#realForward(double[])}.
     */
    @Test
    public void testRealForward()
    {
        final double[] actual = new double[2 * n];
        final double[] expected = new double[2 * n];
        for (int i = 0; i < n; i++) {
            actual[i] = 2. * random.nextDouble() - 1.;
            expected[2 * i] = actual[i];
            expected[2 * i + 1] = 0.;
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
     * This is a test of {@link DoubleFFT_1D#realForward(DoubleLargeArray)}.
     */
    @Test
    public void testRealForwardLarge()
    {
        final DoubleLargeArray actual = new DoubleLargeArray(2 * n);
        final DoubleLargeArray expected = new DoubleLargeArray(2 * n);
        for (int i = 0; i < n; i++) {
            actual.setDouble(i, 2. * random.nextDouble() - 1.);
            expected.setDouble(2 * i, actual.getDouble(i));
            expected.setDouble(2 * i + 1, 0.);
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
            actual.setDouble(n, actual.getDouble(1));
            actual.setDouble(1, 0);
            for (int k = 1; k < m; k++) {
                int idx1 = 2 * n - 2 * k;
                int idx2 = 2 * k;
                actual.setDouble(idx1, actual.getDouble(idx2));
                actual.setDouble(idx1 + 1, -actual.getDouble(idx2 + 1));
            }
        } else {
            for (int k = 1; k < n / 2; k++) {
                int idx1 = 2 * n - 2 * k;
                int idx2 = 2 * k;
                actual.setDouble(idx1, actual.getDouble(idx2));
                actual.setDouble(idx1 + 1, -actual.getDouble(idx2 + 1));
            }
            actual.setDouble(n, actual.getDouble(1));
            actual.setDouble(1, 0);
        }

        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link DoubleFFT_1D#realForward(double[])}.
     */
    @Test
    public void testRealForwardFull()
    {
        final double[] actual = new double[2 * n];
        final double[] expected = new double[2 * n];
        for (int i = 0; i < n; i++) {
            actual[i] = 2. * random.nextDouble() - 1.;
            expected[2 * i] = actual[i];
            expected[2 * i + 1] = 0.;
        }
        fft.complexForward(expected);
        fft.realForwardFull(actual);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link DoubleFFT_1D#realForward(DoubleLargeArray)}.
     */
    @Test
    public void testRealForwardFullLarge()
    {
        final DoubleLargeArray actual = new DoubleLargeArray(2 * n);
        final DoubleLargeArray expected = new DoubleLargeArray(2 * n);
        for (int i = 0; i < n; i++) {
            actual.setDouble(i, 2. * random.nextDouble() - 1.);
            expected.setDouble(2 * i, actual.getDouble(i));
            expected.setDouble(2 * i + 1, 0.);
        }
        fft.complexForward(expected);
        fft.realForwardFull(actual);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link DoubleFFT_1D#realInverseFull(double[], boolean)}
     * , with the second parameter set to <code>true</code>.
     */
    @Test
    public void testRealInverseFullScaled()
    {
        final double[] actual = new double[2 * n];
        final double[] expected = new double[2 * n];
        for (int i = 0; i < n; i++) {
            actual[i] = 2. * random.nextDouble() - 1.;
            expected[2 * i] = actual[i];
            expected[2 * i + 1] = 0.;
        }
        fft.realInverseFull(actual, true);
        fft.complexInverse(expected, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link DoubleFFT_1D#realInverseFull(DoubleLargeArray, boolean)}
     * , with the second parameter set to <code>true</code>.
     */
    @Test
    public void testRealInverseFullScaledLarge()
    {
        final DoubleLargeArray actual = new DoubleLargeArray(2 * n);
        final DoubleLargeArray expected = new DoubleLargeArray(2 * n);
        for (int i = 0; i < n; i++) {
            actual.setDouble(i, 2. * random.nextDouble() - 1.);
            expected.setDouble(2 * i, actual.getDouble(i));
            expected.setDouble(2 * i + 1, 0.);
        }
        fft.realInverseFull(actual, true);
        fft.complexInverse(expected, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link DoubleFFT_1D#realInverseFull(double[], boolean)}
     * , with the second parameter set to <code>false</code>.
     */
    @Test
    public void testRealInverseFullUnscaled()
    {
        final double[] actual = new double[2 * n];
        final double[] expected = new double[2 * n];
        for (int i = 0; i < n; i++) {
            actual[i] = 2. * random.nextDouble() - 1.;
            expected[2 * i] = actual[i];
            expected[2 * i + 1] = 0.;
        }
        fft.realInverseFull(actual, false);
        fft.complexInverse(expected, false);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link DoubleFFT_1D#realInverseFull(DoubleLargeArray, boolean)}
     * , with the second parameter set to <code>false</code>.
     */
    @Test
    public void testRealInverseFullUnscaledLarge()
    {
        final DoubleLargeArray actual = new DoubleLargeArray(2 * n);
        final DoubleLargeArray expected = new DoubleLargeArray(2 * n);
        for (int i = 0; i < n; i++) {
            actual.setDouble(i, 2. * random.nextDouble() - 1.);
            expected.setDouble(2 * i, actual.getDouble(i));
            expected.setDouble(2 * i + 1, 0.);
        }
        fft.realInverseFull(actual, false);
        fft.complexInverse(expected, false);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link DoubleFFT_1D#realInverse(double[], boolean)},
     * with the second parameter set to <code>true</code>.
     */
    @Test
    public void testRealInverseScaled()
    {
        final double[] actual = new double[n];
        final double[] expected = new double[n];
        for (int i = 0; i < n; i++) {
            actual[i] = 2. * random.nextDouble() - 1.;
            expected[i] = actual[i];
        }
        fft.realForward(actual);
        fft.realInverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link DoubleFFT_1D#realInverse(DoubleLargeArray, boolean)},
     * with the second parameter set to <code>true</code>.
     */
    @Test
    public void testRealInverseScaledLarge()
    {
        final DoubleLargeArray actual = new DoubleLargeArray(n);
        final DoubleLargeArray expected = new DoubleLargeArray(n);
        for (int i = 0; i < n; i++) {
            actual.setDouble(i, 2. * random.nextDouble() - 1.);
            expected.setDouble(i, actual.getDouble(i));
        }
        fft.realForward(actual);
        fft.realInverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link DoubleFFT_1D#realInverse(double[], boolean)},
     * with the second parameter set to <code>false</code>.
     */
    @Test
    public void testRealInverseUnscaled()
    {
        final double[] actual = new double[n];
        final double[] expected = new double[n];
        for (int i = 0; i < n; i++) {
            actual[i] = 2. * random.nextDouble() - 1.;
            expected[i] = actual[i];
        }
        fft.realForward(actual);
        fft.realInverse(actual, false);
        double s;
        if (CommonUtils.isPowerOf2(n) && n > 1) {
            s = 2. / (double) n;
        } else {
            s = 1. / (double) n;
        }
        for (int i = 0; i < actual.length; i++) {
            actual[i] = s * actual[i];
        }
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link DoubleFFT_1D#realInverse(DoubleLargeArray, boolean)},
     * with the second parameter set to <code>false</code>.
     */
    @Test
    public void testRealInverseUnscaledLarge()
    {
        final DoubleLargeArray actual = new DoubleLargeArray(n);
        final DoubleLargeArray expected = new DoubleLargeArray(n);
        for (int i = 0; i < n; i++) {
            actual.setDouble(i, 2. * random.nextDouble() - 1.);
            expected.setDouble(i, actual.getDouble(i));
        }
        fft.realForward(actual);
        fft.realInverse(actual, false);

        double s;
        if (CommonUtils.isPowerOf2(n) && n > 1) {
            s = 2. / (double) n;
        } else {
            s = 1. / (double) n;
        }
        for (int i = 0; i < actual.length(); i++) {
            actual.setDouble(i, s * actual.getDouble(i));
        }
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }
}
