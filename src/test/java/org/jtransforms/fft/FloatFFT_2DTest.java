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
import pl.edu.icm.jlargearrays.FloatLargeArray;
import pl.edu.icm.jlargearrays.LargeArray;
import static org.apache.commons.math3.util.FastMath.*;

/**
 *
 * This is a test of the class {@link FloatFFT_2D}. In this test, a very crude
 * 2d FFT method is implemented (see {@link #complexForward(float[][])}),
 * assuming that {@link FloatFFT_1D} has been fully tested and validated. This
 * crude (unoptimized) method is then used to establish <em>expected</em> values
 * of <em>direct</em> Fourier transforms.
 * </p>
 *  
 * For <em>inverse</em> Fourier transforms, the test assumes that the
 * corresponding <em>direct</em> Fourier transform has been tested and
 * validated.
 * </p>
 *  
 * In all cases, the test consists in creating a random array of data, and
 * verifying that expected and actual values of its Fourier transform coincide
 * (L2 norm is zero, within a specified accuracy).
 * </p>
 *
 * @author S&eacute;bastien Brisard
 * @author Piotr Wendykier
 */
@RunWith(value = Parameterized.class)
public class FloatFFT_2DTest
{

    /**
     * Base message of all exceptions.
     */
    public static final String DEFAULT_MESSAGE = "%d-threaded FFT of size %dx%d: ";

    /**
     * The constant value of the seed of the random generator.
     */
    public static final int SEED = 20110602;

    private static final double EPS = pow(10, -3);

    private static final double EPS_UNSCALED = 0.5;

    @Parameters
    public static Collection<Object[]> getParameters()
    {
        final int[] size = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 32,
                            64, 100, 120, 128, 256, 310, 511, 512, 1024};

        final ArrayList<Object[]> parameters = new ArrayList<Object[]>();

        for (int i = 0; i < size.length; i++) {
            for (int j = 0; j < size.length; j++) {
                parameters.add(new Object[]{size[i], size[j], 1, SEED});
                parameters.add(new Object[]{size[i], size[j], 8, SEED});
            }
        }
        return parameters;
    }

    /**
     * Fourier transform of the columns.
     */
    private final FloatFFT_1D cfft;

    /**
     * The object to be tested.
     */
    private final FloatFFT_2D fft;

    /**
     * Number of columns of the data arrays to be Fourier transformed.
     */
    private final int numCols;

    /**
     * Number of rows of the data arrays to be Fourier transformed.
     */
    private final int numRows;

    /**
     * Fourier transform of the rows.
     */
    private final FloatFFT_1D rfft;

    /**
     * For the generation of the data arrays.
     */
    private final Random random;

    /**
     * The number of threads used.
     */
    private final int numThreads;

    /**
     * Creates a new instance of this test.
     *
     * @param numRows
     *                   number of rows
     * @param numColumns
     *                   number of columns
     * @param numThreads
     *                   the number of threads to be used
     * @param seed
     *                   the seed of the random generator
     */
    public FloatFFT_2DTest(final int numRows, final int numColumns,
                           final int numThreads, final long seed)
    {
        this.numRows = numRows;
        this.numCols = numColumns;
        LargeArray.setMaxSizeOf32bitArray(1);
        this.rfft = new FloatFFT_1D(numColumns);
        this.cfft = new FloatFFT_1D(numRows);
        this.fft = new FloatFFT_2D(numRows, numColumns);
        this.random = new Random(seed);
        ConcurrencyUtils.setNumberOfThreads(numThreads);
        CommonUtils.setThreadsBeginN_2D(4);
        this.numThreads = ConcurrencyUtils.getNumberOfThreads();
    }

    /**
     * A crude implementation of 2d complex FFT.
     *
     * @param a
     *          the data to be transformed
     */
    public void complexForward(final float[][] a)
    {
        for (int r = 0; r < numRows; r++) {
            rfft.complexForward(a[r]);
        }
        final float[] buffer = new float[2 * numRows];
        for (int c = 0; c < numCols; c++) {
            for (int r = 0; r < numRows; r++) {
                buffer[2 * r] = a[r][2 * c];
                buffer[2 * r + 1] = a[r][2 * c + 1];
            }
            cfft.complexForward(buffer);
            for (int r = 0; r < numRows; r++) {
                a[r][2 * c] = buffer[2 * r];
                a[r][2 * c + 1] = buffer[2 * r + 1];
            }
        }
    }

    /**
     * A test of {@link FloatFFT_2D#complexForward(float[])}.
     */
    @Test
    public void testComplexForward1dInput()
    {
        final float[] actual = new float[2 * numRows * numCols];
        final float[][] expected0 = new float[numRows][2 * numCols];
        final float[] expected = new float[2 * numRows * numCols];
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < 2 * numCols; c++) {
                final float rnd = random.nextFloat();
                actual[2 * r * numCols + c] = rnd;
                expected0[r][c] = rnd;
            }
        }
        fft.complexForward(actual);
        complexForward(expected0);
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < 2 * numCols; c++) {
                expected[2 * r * numCols + c] = expected0[r][c];
            }
        }
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_2D#complexForward(FloatLargeArray)}.
     */
    @Test
    public void testComplexForwardLarge()
    {
        final FloatLargeArray actual = new FloatLargeArray(2 * numRows * numCols);
        final float[][] expected0 = new float[numRows][2 * numCols];
        final FloatLargeArray expected = new FloatLargeArray(2 * numRows * numCols);
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < 2 * numCols; c++) {
                final float rnd = random.nextFloat();
                actual.setDouble(2 * r * numCols + c, rnd);
                expected0[r][c] = rnd;
            }
        }
        fft.complexForward(actual);
        complexForward(expected0);
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < 2 * numCols; c++) {
                expected.setDouble(2 * r * numCols + c, expected0[r][c]);
            }
        }
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_2D#complexForward(float[][])}.
     */
    @Test
    public void testComplexForward2dInput()
    {
        final float[][] actual = new float[numRows][2 * numCols];
        final float[][] expected = new float[numRows][2 * numCols];
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < 2 * numCols; c++) {
                final float rnd = random.nextFloat();
                actual[r][c] = rnd;
                expected[r][c] = rnd;
            }
        }
        fft.complexForward(actual);
        complexForward(expected);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_2D#complexInverse(float[], boolean)}, with
     * the second parameter set to <code>true</code>.
     */
    @Test
    public void testComplexInverseScaled1dInput()
    {
        final float[] expected = new float[2 * numRows * numCols];
        final float[] actual = new float[2 * numRows * numCols];
        for (int i = 0; i < actual.length; i++) {
            final float rnd = random.nextFloat();
            actual[i] = rnd;
            expected[i] = rnd;
        }
        fft.complexForward(actual);
        fft.complexInverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_2D#complexInverse(FloatLargeArray, boolean)}, with
     * the second parameter set to <code>true</code>.
     */
    @Test
    public void testComplexInverseScaledLarge()
    {

        final FloatLargeArray expected = new FloatLargeArray(2 * numRows * numCols);
        final FloatLargeArray actual = new FloatLargeArray(2 * numRows * numCols);
        for (int i = 0; i < actual.length(); i++) {
            final float rnd = random.nextFloat();
            actual.setDouble(i, rnd);
            expected.setDouble(i, rnd);
        }
        fft.complexForward(actual);
        fft.complexInverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_2D#complexInverse(float[][], boolean)}, with
     * the second parameter set to <code>true</code>.
     */
    @Test
    public void testComplexInverseScaled2dInput()
    {
        final float[][] expected = new float[numRows][2 * numCols];
        final float[][] actual = new float[numRows][2 * numCols];
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < 2 * numCols; c++) {
                final float rnd = random.nextFloat();
                actual[r][c] = rnd;
                expected[r][c] = rnd;
            }
        }
        fft.complexForward(actual);
        fft.complexInverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_2D#complexInverse(float[], boolean)}, with
     * the second parameter set to <code>false</code>.
     */
    @Test
    public void testComplexInverseUnScaled1dInput()
    {
        final float[] expected = new float[2 * numRows * numCols];
        final float[] actual = new float[2 * numRows * numCols];
        for (int i = 0; i < actual.length; i++) {
            final float rnd = random.nextFloat();
            actual[i] = rnd;
            expected[i] = rnd;
        }
        fft.complexForward(actual);
        fft.complexInverse(actual, false);
        final float s = numRows * numCols;
        for (int i = 0; i < actual.length; i++) {
            actual[i] = actual[i] / s;
        }
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_2D#complexInverse(FloatLargeArray, boolean)}, with
     * the second parameter set to <code>false</code>.
     */
    @Test
    public void testComplexInverseUnScaledLarge()
    {
        final FloatLargeArray expected = new FloatLargeArray(2 * numRows * numCols);
        final FloatLargeArray actual = new FloatLargeArray(2 * numRows * numCols);
        for (int i = 0; i < actual.length(); i++) {
            final float rnd = random.nextFloat();
            actual.setDouble(i, rnd);
            expected.setDouble(i, rnd);
        }
        fft.complexForward(actual);
        fft.complexInverse(actual, false);
        final float s = numRows * numCols;
        for (int i = 0; i < actual.length(); i++) {
            actual.setDouble(i, actual.getDouble(i) / s);
        }
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_2D#complexInverse(float[][], boolean)}, with
     * the second parameter set to <code>false</code>.
     */
    @Test
    public void testComplexInverseUnScaled2dInput()
    {
        final float[][] expected = new float[numRows][2 * numCols];
        final float[][] actual = new float[numRows][2 * numCols];
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < 2 * numCols; c++) {
                final float rnd = random.nextFloat();
                expected[r][c] = rnd;
                actual[r][c] = rnd;
            }
        }
        fft.complexForward(actual);
        fft.complexInverse(actual, false);
        final float s = numRows * numCols;
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < 2 * numCols; c++) {
                actual[r][c] = actual[r][c] / s;
            }
        }
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    private static void fillSymmetric(final float[] a, int rows, int columns)
    {
        final int twon2 = 2 * columns;
        int idx1, idx2, idx3, idx4;
        int n1d2 = rows / 2;

        for (int r = (rows - 1); r >= 1; r--) {
            idx1 = r * columns;
            idx2 = 2 * idx1;
            for (int c = 0; c < columns; c += 2) {
                a[idx2 + c] = a[idx1 + c];
                a[idx1 + c] = 0;
                a[idx2 + c + 1] = a[idx1 + c + 1];
                a[idx1 + c + 1] = 0;
            }
        }

        for (int r = 1; r < n1d2; r++) {
            idx2 = r * twon2;
            idx3 = (rows - r) * twon2;
            a[idx2 + columns] = a[idx3 + 1];
            a[idx2 + columns + 1] = -a[idx3];
        }

        for (int r = 1; r < n1d2; r++) {
            idx2 = r * twon2;
            idx3 = (rows - r + 1) * twon2;
            for (int c = columns + 2; c < twon2; c += 2) {
                a[idx2 + c] = a[idx3 - c];
                a[idx2 + c + 1] = -a[idx3 - c + 1];

            }
        }
        for (int r = 0; r <= rows / 2; r++) {
            idx1 = r * twon2;
            idx4 = ((rows - r) % rows) * twon2;
            for (int c = 0; c < twon2; c += 2) {
                idx2 = idx1 + c;
                idx3 = idx4 + (twon2 - c) % twon2;
                a[idx3] = a[idx2];
                a[idx3 + 1] = -a[idx2 + 1];
            }
        }
        a[columns] = -a[1];
        a[1] = 0;
        idx1 = n1d2 * twon2;
        a[idx1 + columns] = -a[idx1 + 1];
        a[idx1 + 1] = 0;
        a[idx1 + columns + 1] = 0;
    }

    /**
     * A test of {@link FloatFFT_2D#realForward(float[])}.
     */
    @Test
    public void testRealForward1dInput()
    {
        if (!CommonUtils.isPowerOf2(numRows)) {
            return;
        }
        if (!CommonUtils.isPowerOf2(numCols)) {
            return;
        }
        final float[] actual = new float[2 * numRows * numCols];
        final float[] expected = new float[numRows * 2 * numCols];
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                final float rnd = random.nextFloat();
                actual[r * numCols + c] = rnd;
                expected[r * 2 * numCols + 2 * c] = rnd;
            }
        }
        fft.realForward(actual);
        fft.complexForward(expected);
        fillSymmetric(actual, numRows, numCols);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    private static void fillSymmetric(final FloatLargeArray a, int rowsl, int columnsl)
    {
        final long twon2 = 2 * columnsl;
        long idx1, idx2, idx3, idx4;
        long n1d2 = rowsl / 2;

        for (long r = (rowsl - 1); r >= 1; r--) {
            idx1 = r * columnsl;
            idx2 = 2 * idx1;
            for (long c = 0; c < columnsl; c += 2) {
                a.setDouble(idx2 + c, a.getDouble(idx1 + c));
                a.setDouble(idx1 + c, 0);
                a.setDouble(idx2 + c + 1, a.getDouble(idx1 + c + 1));
                a.setDouble(idx1 + c + 1, 0);
            }
        }

        for (long r = 1; r < n1d2; r++) {
            idx2 = r * twon2;
            idx3 = (rowsl - r) * twon2;
            a.setDouble(idx2 + columnsl, a.getDouble(idx3 + 1));
            a.setDouble(idx2 + columnsl + 1, -a.getDouble(idx3));
        }

        for (long r = 1; r < n1d2; r++) {
            idx2 = r * twon2;
            idx3 = (rowsl - r + 1) * twon2;
            for (long c = columnsl + 2; c < twon2; c += 2) {
                a.setDouble(idx2 + c, a.getDouble(idx3 - c));
                a.setDouble(idx2 + c + 1, -a.getDouble(idx3 - c + 1));

            }
        }
        for (long r = 0; r <= rowsl / 2; r++) {
            idx1 = r * twon2;
            idx4 = ((rowsl - r) % rowsl) * twon2;
            for (long c = 0; c < twon2; c += 2) {
                idx2 = idx1 + c;
                idx3 = idx4 + (twon2 - c) % twon2;
                a.setDouble(idx3, a.getDouble(idx2));
                a.setDouble(idx3 + 1, -a.getDouble(idx2 + 1));
            }
        }

        a.setDouble(columnsl, -a.getDouble(1));
        a.setDouble(1, 0);
        idx1 = n1d2 * twon2;
        a.setDouble(idx1 + columnsl, -a.getDouble(idx1 + 1));
        a.setDouble(idx1 + 1, 0);
        a.setDouble(idx1 + columnsl + 1, 0);
    }

    /**
     * A test of {@link FloatFFT_2D#realForward(FloatLargeArray)}.
     */
    @Test
    public void testRealForwardLarge()
    {
        if (!CommonUtils.isPowerOf2(numRows)) {
            return;
        }
        if (!CommonUtils.isPowerOf2(numCols)) {
            return;
        }
        final FloatLargeArray actual = new FloatLargeArray(2 * numRows * numCols);
        final FloatLargeArray expected = new FloatLargeArray(numRows * 2 * numCols);
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                final float rnd = random.nextFloat();
                actual.setDouble(r * numCols + c, rnd);
                expected.setDouble(r * 2 * numCols + 2 * c, rnd);
            }
        }
        fft.realForward(actual);
        fft.complexForward(expected);
        fillSymmetric(actual, numRows, numCols);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    private void fillSymmetric(final float[][] a, int rows, int columns)
    {
        final int newn2 = 2 * columns;
        int n1d2 = rows / 2;

        for (int r = 1; r < n1d2; r++) {
            int idx1 = rows - r;
            a[r][columns] = a[idx1][1];
            a[r][columns + 1] = -a[idx1][0];
        }
        for (int r = 1; r < n1d2; r++) {
            int idx1 = rows - r;
            for (int c = columns + 2; c < newn2; c += 2) {
                int idx2 = newn2 - c;
                a[r][c] = a[idx1][idx2];
                a[r][c + 1] = -a[idx1][idx2 + 1];
            }
        }
        for (int r = 0; r <= rows / 2; r++) {
            int idx1 = (rows - r) % rows;
            for (int c = 0; c < newn2; c += 2) {
                int idx2 = (newn2 - c) % newn2;
                a[idx1][idx2] = a[r][c];
                a[idx1][idx2 + 1] = -a[r][c + 1];
            }
        }
        a[0][columns] = -a[0][1];
        a[0][1] = 0;
        a[n1d2][columns] = -a[n1d2][1];
        a[n1d2][1] = 0;
        a[n1d2][columns + 1] = 0;
    }

    /**
     * A test of {@link FloatFFT_2D#realForward(float[][])}.
     */
    @Test
    public void testRealForward2dInput()
    {
        if (!CommonUtils.isPowerOf2(numRows)) {
            return;
        }
        if (!CommonUtils.isPowerOf2(numCols)) {
            return;
        }
        final float[][] actual = new float[numRows][2 * numCols];
        final float[][] expected = new float[numRows][2 * numCols];
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                final float rnd = random.nextFloat();
                actual[r][c] = rnd;
                expected[r][2 * c] = rnd;
            }
        }
        fft.realForward(actual);
        complexForward(expected);
        fillSymmetric(actual, numRows, numCols);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);

    }

    /**
     * A test of {@link FloatFFT_2D#realForwardFull(float[])}.
     */
    @Test
    public void testRealForwardFull1dInput()
    {
        final float[] actual = new float[2 * numRows * numCols];
        final float[][] expected0 = new float[numRows][2 * numCols];
        final float[] expected = new float[numRows * 2 * numCols];
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                final float rnd = random.nextFloat();
                actual[r * numCols + c] = rnd;
                expected0[r][2 * c] = rnd;
            }
        }
        fft.realForwardFull(actual);
        complexForward(expected0);
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < 2 * numCols; c++) {
                expected[2 * r * numCols + c] = expected0[r][c];
            }
        }
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_2D#realForwardFull(FloatLargeArray)}.
     */
    @Test
    public void testRealForwardFullLarge()
    {
        final FloatLargeArray actual = new FloatLargeArray(2 * numRows * numCols);
        final float[][] expected0 = new float[numRows][2 * numCols];
        final FloatLargeArray expected = new FloatLargeArray(numRows * 2 * numCols);
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                final float rnd = random.nextFloat();
                actual.setDouble(r * numCols + c, rnd);
                expected0[r][2 * c] = rnd;
            }
        }
        fft.realForwardFull(actual);
        complexForward(expected0);
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < 2 * numCols; c++) {
                expected.setDouble(2 * r * numCols + c, expected0[r][c]);
            }
        }
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_2D#realForwardFull(float[][])}.
     */
    @Test
    public void testRealForwardFull2dInput()
    {
        final float[][] actual = new float[numRows][2 * numCols];
        final float[][] expected = new float[numRows][2 * numCols];
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                final float rnd = random.nextFloat();
                actual[r][c] = rnd;
                expected[r][2 * c] = rnd;
            }
        }
        fft.realForwardFull(actual);
        complexForward(expected);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_2D#realInverseFull(float[], boolean)}, with
     * the second parameter set to <code>true</code>.
     */
    @Test
    public void testRealInverseFullScaled1dInput()
    {
        final float[] actual = new float[2 * numRows * numCols];
        final float[] expected = new float[2 * numRows * numCols];
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                final float rnd = random.nextFloat();
                final int index = r * numCols + c;
                actual[index] = rnd;
                expected[2 * index] = rnd;
            }
        }
        // TODO If the two following lines are permuted, this causes an array
        // index out of bounds exception.
        fft.complexInverse(expected, true);
        fft.realInverseFull(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_2D#realInverseFull(FloatLargeArray, boolean)}, with
     * the second parameter set to <code>true</code>.
     */
    @Test
    public void testRealInverseFullScaledLarge()
    {
        final FloatLargeArray actual = new FloatLargeArray(2 * numRows * numCols);
        final FloatLargeArray expected = new FloatLargeArray(2 * numRows * numCols);
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                final float rnd = random.nextFloat();
                final int index = r * numCols + c;
                actual.setDouble(index, rnd);
                expected.setDouble(2 * index, rnd);
            }
        }
        // TODO If the two following lines are permuted, this causes an array
        // index out of bounds exception.
        fft.complexInverse(expected, true);
        fft.realInverseFull(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_2D#realInverseFull(float[][], boolean)}, with
     * the second parameter set to <code>true</code>.
     */
    @Test
    public void testRealInverseFullScaled2dInput()
    {
        final float[][] actual = new float[numRows][2 * numCols];
        final float[][] expected = new float[numRows][2 * numCols];
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                final float rnd = random.nextFloat();
                actual[r][c] = rnd;
                expected[r][2 * c] = rnd;
            }
        }
        fft.realInverseFull(actual, true);
        fft.complexInverse(expected, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_2D#realInverseFull(float[], boolean)}, with
     * the second parameter set to <code>false</code>.
     */
    @Test
    public void testRealInverseFullUnscaled1dInput()
    {
        final float[] actual = new float[2 * numRows * numCols];
        final float[] expected = new float[2 * numRows * numCols];
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                final float rnd = random.nextFloat();
                final int index = r * numCols + c;
                actual[index] = rnd;
                expected[2 * index] = rnd;
            }
        }
        // TODO If the two following lines are permuted, this causes an array
        // index out of bounds exception.
        fft.complexInverse(expected, false);
        fft.realInverseFull(actual, false);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS_UNSCALED);
    }

    /**
     * A test of {@link FloatFFT_2D#realInverseFull(FloatLargeArray, boolean)}, with
     * the second parameter set to <code>false</code>.
     */
    @Test
    public void testRealInverseFullUnscaledLarge()
    {
        final FloatLargeArray actual = new FloatLargeArray(2 * numRows * numCols);
        final FloatLargeArray expected = new FloatLargeArray(2 * numRows * numCols);
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                final float rnd = random.nextFloat();
                final int index = r * numCols + c;
                actual.set(index, rnd);
                expected.setDouble(2 * index, rnd);
            }
        }
        // TODO If the two following lines are permuted, this causes an array
        // index out of bounds exception.
        fft.complexInverse(expected, false);
        fft.realInverseFull(actual, false);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS_UNSCALED);
    }

    /**
     * A test of {@link FloatFFT_2D#realInverseFull(float[][], boolean)}, with
     * the second parameter set to <code>false</code>.
     */
    @Test
    public void testRealInverseFullUnscaled2dInput()
    {
        final float[][] actual = new float[numRows][2 * numCols];
        final float[][] expected = new float[numRows][2 * numCols];
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                final float rnd = random.nextFloat();
                actual[r][c] = rnd;
                expected[r][2 * c] = rnd;
            }
        }
        fft.realInverseFull(actual, false);
        fft.complexInverse(expected, false);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS_UNSCALED);
    }

    /**
     * A test of {@link FloatFFT_2D#realInverse(float[], boolean)}, with the
     * second parameter set to <code>true</code>.
     */
    @Test
    public void testRealInverseScaled1dInput()
    {
        if (!CommonUtils.isPowerOf2(numRows)) {
            return;
        }
        if (!CommonUtils.isPowerOf2(numCols)) {
            return;
        }
        final float[] actual = new float[numRows * numCols];
        final float[] expected = new float[actual.length];
        for (int i = 0; i < actual.length; i++) {
            final float rnd = random.nextFloat();
            actual[i] = rnd;
            expected[i] = rnd;
        }
        fft.realForward(actual);
        fft.realInverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_2D#realInverse(FloatLargeArray, boolean)}, with the
     * second parameter set to <code>true</code>.
     */
    @Test
    public void testRealInverseScaledLarge()
    {
        if (!CommonUtils.isPowerOf2(numRows)) {
            return;
        }
        if (!CommonUtils.isPowerOf2(numCols)) {
            return;
        }
        final FloatLargeArray actual = new FloatLargeArray(numRows * numCols);
        final FloatLargeArray expected = new FloatLargeArray(actual.length());
        for (int i = 0; i < actual.length(); i++) {
            final float rnd = random.nextFloat();
            actual.setDouble(i, rnd);
            expected.setDouble(i, rnd);
        }
        fft.realForward(actual);
        fft.realInverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_2D#realInverse(float[][], boolean)}, with the
     * second parameter set to <code>true</code>.
     */
    @Test
    public void testRealInverseScaled2dInput()
    {
        if (!CommonUtils.isPowerOf2(numRows)) {
            return;
        }
        if (!CommonUtils.isPowerOf2(numCols)) {
            return;
        }
        final float[][] actual = new float[numRows][numCols];
        final float[][] expected = new float[numRows][numCols];
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                final float rnd = random.nextFloat();
                actual[r][c] = rnd;
                expected[r][c] = rnd;
            }
        }
        fft.realForward(actual);
        fft.realInverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }
}
