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
 * This is a test of the class {@link FloatFFT_3D}. In this test, a very crude
 * 3d FFT method is implemented (see {@link #complexForward(float[][][])}),
 * assuming that {@link FloatFFT_1D} and {@link FloatFFT_2D} have been fully
 * tested and validated. This crude (unoptimized) method is then used to
 * establish <em>expected</em> values of <em>direct</em> Fourier transforms.
 * </p>
 *  
 * For <em>inverse</em> Fourier transforms, the test assumes that the
 * corresponding <em>direct</em> Fourier transform has been tested and
 * validated.
 * </p>
 *  
 * In all cases, the test consists in creating a random array of data, and
 * verifying that expected and actual values of its Fourier transform coincide
 * within a specified accuracy.
 * </p>
 *
 * @author S&eacute;bastien Brisard
 * @author Piotr Wendykier
 */
@RunWith(value = Parameterized.class)
public class FloatFFT_3DTest
{

    /**
     * Base message of all exceptions.
     */
    public static final String DEFAULT_MESSAGE = "%d-threaded FFT of size %dx%dx%d: ";

    /**
     * The constant value of the seed of the random generator.
     */
    public static final int SEED = 20110625;

    private static final double EPS = pow(10, -3);

    private static final double EPS_UNSCALED = 0.5;

    @Parameters
    public static Collection<Object[]> getParameters()
    {
        final int[] size = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 32,
                            64, 100, 128};

        final ArrayList<Object[]> parameters = new ArrayList<Object[]>();

        for (int i = 0; i < size.length; i++) {
            for (int j = 0; j < size.length; j++) {
                for (int k = 0; k < size.length; k++) {
                    parameters.add(new Object[]{size[i], size[j], size[k], 1,
                                                SEED});
                    parameters.add(new Object[]{size[i], size[j], size[k], 8,
                                                SEED});
                }
            }
        }
        return parameters;
    }

    /**
     * The object to be tested.
     */
    private final FloatFFT_3D fft;

    /**
     * Number of columns of the data arrays to be Fourier transformed.
     */
    private final int numCols;

    /**
     * Number of rows of the data arrays to be Fourier transformed.
     */
    private final int numRows;

    /**
     * Number of slices of the data arrays to be Fourier transformed.
     */
    private final int numSlices;

    /**
     * For the generation of the data arrays.
     */
    private final Random random;

    /**
     * Fourier transform of the slices.
     */
    private final FloatFFT_2D sfft;

    /**
     * Fourier transform in the first direction (perpendicular to slices).
     */
    private final FloatFFT_1D xfft;

    /**
     * The number of threads used.
     */
    private final int numThreads;

    /**
     * Creates a new instance of this test.
     *
     * @param numSlices
     *                   number of slices
     * @param numRows
     *                   number of rows
     * @param numColumns
     *                   number of columns
     * @param numThreads
     *                   the number of threads to be used
     * @param seed
     *                   the seed of the random generator
     */
    public FloatFFT_3DTest(final int numSlices, final int numRows,
                           final int numColumns, final int numThreads, final long seed)
    {
        this.numSlices = numSlices;
        this.numRows = numRows;
        this.numCols = numColumns;
        LargeArray.setMaxSizeOf32bitArray(1);
        this.fft = new FloatFFT_3D(numSlices, numRows, numColumns);
        this.xfft = new FloatFFT_1D(numSlices);
        this.sfft = new FloatFFT_2D(numRows, numColumns);
        this.random = new Random(seed);
        ConcurrencyUtils.setNumberOfThreads(numThreads);
        CommonUtils.setThreadsBeginN_3D(4);
        this.numThreads = ConcurrencyUtils.getNumberOfThreads();
    }

    /**
     * A crude implementation of 3d complex FFT.
     *
     * @param a
     *          the data to be transformed
     */
    public void complexForward(final float[][][] a)
    {
        for (int s = 0; s < numSlices; s++) {
            sfft.complexForward(a[s]);
        }
        final float[] buffer = new float[2 * numSlices];
        for (int c = 0; c < numCols; c++) {
            for (int r = 0; r < numRows; r++) {
                for (int s = 0; s < numSlices; s++) {
                    buffer[2 * s] = a[s][r][2 * c];
                    buffer[2 * s + 1] = a[s][r][2 * c + 1];
                }
                xfft.complexForward(buffer);
                for (int s = 0; s < numSlices; s++) {
                    a[s][r][2 * c] = buffer[2 * s];
                    a[s][r][2 * c + 1] = buffer[2 * s + 1];
                }
            }
        }
    }

    /**
     * A test of {@link FloatFFT_3D#complexForward(float[])}.
     */
    @Test
    public void testComplexForward1dInput()
    {
        final float[] actual = new float[2 * numSlices * numRows * numCols];
        final float[][][] expected0 = new float[numSlices][numRows][2 * numCols];
        final float[] expected = new float[2 * numSlices * numRows * numCols];
        for (int s = 0; s < numSlices; s++) {
            for (int r = 0; r < numRows; r++) {
                for (int c = 0; c < 2 * numCols; c++) {
                    final int index = 2 * numCols * (r + numRows * s) + c;
                    final float rnd = random.nextFloat();
                    actual[index] = rnd;
                    expected0[s][r][c] = rnd;
                }
            }
        }
        fft.complexForward(actual);
        complexForward(expected0);
        for (int s = 0; s < numSlices; s++) {
            for (int r = 0; r < numRows; r++) {
                for (int c = 0; c < 2 * numCols; c++) {
                    expected[s * 2 * numRows * numCols + r * 2 * numCols + c] = expected0[s][r][c];
                }
            }
        }
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numSlices, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_3D#complexForward(FloatLargeArray)}.
     */
    @Test
    public void testComplexForwardLarge()
    {
        final FloatLargeArray actual = new FloatLargeArray(2 * numSlices * numRows * numCols);
        final float[][][] expected0 = new float[numSlices][numRows][2 * numCols];
        final FloatLargeArray expected = new FloatLargeArray(2 * numSlices * numRows * numCols);
        for (int s = 0; s < numSlices; s++) {
            for (int r = 0; r < numRows; r++) {
                for (int c = 0; c < 2 * numCols; c++) {
                    final int index = 2 * numCols * (r + numRows * s) + c;
                    final float rnd = random.nextFloat();
                    actual.setFloat(index, rnd);
                    expected0[s][r][c] = rnd;
                }
            }
        }
        fft.complexForward(actual);
        complexForward(expected0);
        for (int s = 0; s < numSlices; s++) {
            for (int r = 0; r < numRows; r++) {
                for (int c = 0; c < 2 * numCols; c++) {
                    expected.setFloat(s * 2 * numRows * numCols + r * 2 * numCols + c, expected0[s][r][c]);
                }
            }
        }
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numSlices, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_3D#complexForward(float[][][])}.
     */
    @Test
    public void testComplexForward3dInput()
    {
        final float[][][] actual = new float[numSlices][numRows][2 * numCols];
        final float[][][] expected = new float[numSlices][numRows][2 * numCols];
        for (int s = 0; s < numSlices; s++) {
            for (int r = 0; r < numRows; r++) {
                for (int c = 0; c < 2 * numCols; c++) {
                    final float rnd = random.nextFloat();
                    actual[s][r][c] = rnd;
                    expected[s][r][c] = rnd;
                }
            }
        }
        fft.complexForward(actual);
        complexForward(expected);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numSlices, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_3D#complexInverse(float[], boolean)}, with
     * the second parameter set to <code>true</code>.
     */
    @Test
    public void testComplexInverseScaled1dInput()
    {
        final float[] expected = new float[2 * numSlices * numRows * numCols];
        final float[] actual = new float[2 * numSlices * numRows * numCols];
        for (int i = 0; i < actual.length; i++) {
            final float rnd = random.nextFloat();
            actual[i] = rnd;
            expected[i] = rnd;
        }
        fft.complexForward(actual);
        fft.complexInverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numSlices, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_3D#complexInverse(FloatLargeArray, boolean)}, with
     * the second parameter set to <code>true</code>.
     */
    @Test
    public void testComplexInverseScaledLarge()
    {
        final FloatLargeArray expected = new FloatLargeArray(2 * numSlices * numRows * numCols);
        final FloatLargeArray actual = new FloatLargeArray(2 * numSlices * numRows * numCols);
        for (int i = 0; i < actual.length(); i++) {
            final float rnd = random.nextFloat();
            actual.setFloat(i, rnd);
            expected.setFloat(i, rnd);
        }
        fft.complexForward(actual);
        fft.complexInverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numSlices, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_3D#complexInverse(float[][][], boolean)},
     * with the second parameter set to <code>true</code>.
     */
    @Test
    public void testComplexInverseScaled3dInput()
    {
        final float[][][] expected = new float[numSlices][numRows][2 * numCols];
        final float[][][] actual = new float[numSlices][numRows][2 * numCols];
        for (int s = 0; s < numSlices; s++) {
            for (int r = 0; r < numRows; r++) {
                for (int c = 0; c < 2 * numCols; c++) {
                    final float rnd = random.nextFloat();
                    actual[s][r][c] = rnd;
                    expected[s][r][c] = rnd;
                }
            }
        }
        fft.complexForward(actual);
        fft.complexInverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numSlices, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_3D#complexInverse(float[], boolean)}, with
     * the second parameter set to <code>false</code>.
     */
    @Test
    public void testComplexInverseUnscaled1dInput()
    {
        final float[] expected = new float[2 * numSlices * numRows * numCols];
        final float[] actual = new float[2 * numSlices * numRows * numCols];
        for (int i = 0; i < actual.length; i++) {
            final float rnd = random.nextFloat();
            actual[i] = rnd;
            expected[i] = rnd;
        }
        fft.complexForward(actual);
        fft.complexInverse(actual, false);
        final float scaling = numSlices * numRows * numCols;
        for (int i = 0; i < actual.length; i++) {
            expected[i] = scaling * expected[i];
        }
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numSlices, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS_UNSCALED);
    }

    /**
     * A test of {@link FloatFFT_3D#complexInverse(FloatLargeArray, boolean)}, with
     * the second parameter set to <code>false</code>.
     */
    @Test
    public void testComplexInverseUnscaledLarge()
    {
        final FloatLargeArray expected = new FloatLargeArray(2 * numSlices * numRows * numCols);
        final FloatLargeArray actual = new FloatLargeArray(2 * numSlices * numRows * numCols);
        for (int i = 0; i < actual.length(); i++) {
            final float rnd = random.nextFloat();
            actual.setFloat(i, rnd);
            expected.setFloat(i, rnd);
        }
        fft.complexForward(actual);
        fft.complexInverse(actual, false);
        final float scaling = numSlices * numRows * numCols;
        for (int i = 0; i < actual.length(); i++) {
            expected.setFloat(i, scaling * expected.getFloat(i));
        }
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numSlices, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS_UNSCALED);
    }

    /**
     * A test of {@link FloatFFT_3D#complexInverse(float[][][], boolean)},
     * with the second parameter set to <code>false</code>.
     */
    @Test
    public void testComplexInverseUnscaled3dInput()
    {
        final float[][][] expected = new float[numSlices][numRows][2 * numCols];
        final float[][][] actual = new float[numSlices][numRows][2 * numCols];
        for (int s = 0; s < numSlices; s++) {
            for (int r = 0; r < numRows; r++) {
                for (int c = 0; c < 2 * numCols; c++) {
                    final float rnd = random.nextFloat();
                    actual[s][r][c] = rnd;
                    expected[s][r][c] = rnd;
                }
            }
        }
        fft.complexForward(actual);
        fft.complexInverse(actual, false);
        final float scaling = numSlices * numRows * numCols;
        for (int s = 0; s < numSlices; s++) {
            for (int r = 0; r < numRows; r++) {
                for (int c = 0; c < 2 * numCols; c++) {
                    expected[s][r][c] = scaling * expected[s][r][c];
                }
            }
        }
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numSlices, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS_UNSCALED);
    }

    private void fillSymmetric(final float[] a, int slices, int rows, int columns)
    {
        final int twon3 = 2 * columns;
        final int n2d2 = rows / 2;
        int n1d2 = slices / 2;
        int sliceStride = rows * columns;
        int rowStride = columns;

        final int twoSliceStride = rows * twon3;
        final int twoRowStride = twon3;

        int idx1, idx2, idx3, idx4, idx5, idx6;

        for (int s = (slices - 1); s >= 1; s--) {
            idx3 = s * sliceStride;
            idx4 = 2 * idx3;
            for (int r = 0; r < rows; r++) {
                idx5 = r * rowStride;
                idx6 = 2 * idx5;
                for (int c = 0; c < columns; c += 2) {
                    idx1 = idx3 + idx5 + c;
                    idx2 = idx4 + idx6 + c;
                    a[idx2] = a[idx1];
                    a[idx1] = 0;
                    idx1++;
                    idx2++;
                    a[idx2] = a[idx1];
                    a[idx1] = 0;
                }
            }
        }

        for (int r = 1; r < rows; r++) {
            idx3 = (rows - r) * rowStride;
            idx4 = (rows - r) * twoRowStride;
            for (int c = 0; c < columns; c += 2) {
                idx1 = idx3 + c;
                idx2 = idx4 + c;
                a[idx2] = a[idx1];
                a[idx1] = 0;
                idx1++;
                idx2++;
                a[idx2] = a[idx1];
                a[idx1] = 0;
            }
        }

        // -----------------------------------------------
        for (int s = 0; s < slices; s++) {
            idx3 = ((slices - s) % slices) * twoSliceStride;
            idx5 = s * twoSliceStride;
            for (int r = 0; r < rows; r++) {
                idx4 = ((rows - r) % rows) * twoRowStride;
                idx6 = r * twoRowStride;
                for (int c = 1; c < columns; c += 2) {
                    idx1 = idx3 + idx4 + twon3 - c;
                    idx2 = idx5 + idx6 + c;
                    a[idx1] = -a[idx2 + 2];
                    a[idx1 - 1] = a[idx2 + 1];
                }
            }
        }

        // ---------------------------------------------
        for (int s = 0; s < slices; s++) {
            idx5 = ((slices - s) % slices) * twoSliceStride;
            idx6 = s * twoSliceStride;
            for (int r = 1; r < n2d2; r++) {
                idx4 = idx6 + (rows - r) * twoRowStride;
                idx1 = idx5 + r * twoRowStride + columns;
                idx2 = idx4 + columns;
                idx3 = idx4 + 1;
                a[idx1] = a[idx3];
                a[idx2] = a[idx3];
                a[idx1 + 1] = -a[idx4];
                a[idx2 + 1] = a[idx4];

            }
        }

        for (int s = 0; s < slices; s++) {
            idx3 = ((slices - s) % slices) * twoSliceStride;
            idx4 = s * twoSliceStride;
            for (int r = 1; r < n2d2; r++) {
                idx1 = idx3 + (rows - r) * twoRowStride;
                idx2 = idx4 + r * twoRowStride;
                a[idx1] = a[idx2];
                a[idx1 + 1] = -a[idx2 + 1];

            }
        }

        // ----------------------------------------------------------
        for (int s = 1; s < n1d2; s++) {
            idx1 = s * twoSliceStride;
            idx2 = (slices - s) * twoSliceStride;
            idx3 = n2d2 * twoRowStride;
            idx4 = idx1 + idx3;
            idx5 = idx2 + idx3;
            a[idx1 + columns] = a[idx2 + 1];
            a[idx2 + columns] = a[idx2 + 1];
            a[idx1 + columns + 1] = -a[idx2];
            a[idx2 + columns + 1] = a[idx2];
            a[idx4 + columns] = a[idx5 + 1];
            a[idx5 + columns] = a[idx5 + 1];
            a[idx4 + columns + 1] = -a[idx5];
            a[idx5 + columns + 1] = a[idx5];
            a[idx2] = a[idx1];
            a[idx2 + 1] = -a[idx1 + 1];
            a[idx5] = a[idx4];
            a[idx5 + 1] = -a[idx4 + 1];

        }

        // ----------------------------------------
        a[columns] = a[1];
        a[1] = 0;
        idx1 = n2d2 * twoRowStride;
        idx2 = n1d2 * twoSliceStride;
        idx3 = idx1 + idx2;
        a[idx1 + columns] = a[idx1 + 1];
        a[idx1 + 1] = 0;
        a[idx2 + columns] = a[idx2 + 1];
        a[idx2 + 1] = 0;
        a[idx3 + columns] = a[idx3 + 1];
        a[idx3 + 1] = 0;
        a[idx2 + columns + 1] = 0;
        a[idx3 + columns + 1] = 0;
    }

    /**
     * A test of {@link FloatFFT_3D#realForward(float[])}.
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
        if (!CommonUtils.isPowerOf2(numSlices)) {
            return;
        }
        int index;
        final float[] actual = new float[numSlices * numRows * 2 * numCols];
        final float[] expected = new float[numSlices * numRows * 2 * numCols];
        for (int s = 0; s < numSlices; s++) {
            for (int r = 0; r < numRows; r++) {
                for (int c = 0; c < numCols; c++) {
                    index = c + numCols * (r + numRows * s);
                    final float rnd = random.nextFloat();
                    actual[index] = rnd;
                    expected[s * 2 * numRows * numCols + r * 2 * numCols + 2 * c] = rnd;
                }
            }
        }
        fft.realForward(actual);
        fft.complexForward(expected);
        fillSymmetric(actual, numSlices, numRows, numCols);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numSlices, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    private void fillSymmetric(final FloatLargeArray a, long slicesl, int rowsl, long columnsl)
    {
        final long twon3 = 2 * columnsl;
        final long n2d2 = rowsl / 2;
        long n1d2 = slicesl / 2;

        final long twoSliceStride = rowsl * twon3;
        final long twoRowStride = twon3;

        long idx1, idx2, idx3, idx4, idx5, idx6;

        final long sliceStridel = rowsl * columnsl;
        final long rowStridel = columnsl;

        for (long s = (slicesl - 1); s >= 1; s--) {
            idx3 = s * sliceStridel;
            idx4 = 2 * idx3;
            for (long r = 0; r < rowsl; r++) {
                idx5 = r * rowStridel;
                idx6 = 2 * idx5;
                for (long c = 0; c < columnsl; c += 2) {
                    idx1 = idx3 + idx5 + c;
                    idx2 = idx4 + idx6 + c;
                    a.setFloat(idx2, a.getFloat(idx1));
                    a.setFloat(idx1, 0);
                    idx1++;
                    idx2++;
                    a.setFloat(idx2, a.getFloat(idx1));
                    a.setFloat(idx1, 0);
                }
            }
        }

        for (long r = 1; r < rowsl; r++) {
            idx3 = (rowsl - r) * rowStridel;
            idx4 = (rowsl - r) * twoRowStride;
            for (long c = 0; c < columnsl; c += 2) {
                idx1 = idx3 + c;
                idx2 = idx4 + c;
                a.setFloat(idx2, a.getFloat(idx1));
                a.setFloat(idx1, 0);
                idx1++;
                idx2++;
                a.setFloat(idx2, a.getFloat(idx1));
                a.setFloat(idx1, 0);
            }
        }

        // -----------------------------------------------
        for (long s = 0; s < slicesl; s++) {
            idx3 = ((slicesl - s) % slicesl) * twoSliceStride;
            idx5 = s * twoSliceStride;
            for (long r = 0; r < rowsl; r++) {
                idx4 = ((rowsl - r) % rowsl) * twoRowStride;
                idx6 = r * twoRowStride;
                for (long c = 1; c < columnsl; c += 2) {
                    idx1 = idx3 + idx4 + twon3 - c;
                    idx2 = idx5 + idx6 + c;
                    a.setFloat(idx1, -a.getFloat(idx2 + 2));
                    a.setFloat(idx1 - 1, a.getFloat(idx2 + 1));
                }
            }
        }

        // ---------------------------------------------
        for (long s = 0; s < slicesl; s++) {
            idx5 = ((slicesl - s) % slicesl) * twoSliceStride;
            idx6 = s * twoSliceStride;
            for (long r = 1; r < n2d2; r++) {
                idx4 = idx6 + (rowsl - r) * twoRowStride;
                idx1 = idx5 + r * twoRowStride + columnsl;
                idx2 = idx4 + columnsl;
                idx3 = idx4 + 1;
                a.setFloat(idx1, a.getFloat(idx3));
                a.setFloat(idx2, a.getFloat(idx3));
                a.setFloat(idx1 + 1, -a.getFloat(idx4));
                a.setFloat(idx2 + 1, a.getFloat(idx4));

            }
        }

        for (long s = 0; s < slicesl; s++) {
            idx3 = ((slicesl - s) % slicesl) * twoSliceStride;
            idx4 = s * twoSliceStride;
            for (long r = 1; r < n2d2; r++) {
                idx1 = idx3 + (rowsl - r) * twoRowStride;
                idx2 = idx4 + r * twoRowStride;
                a.setFloat(idx1, a.getFloat(idx2));
                a.setFloat(idx1 + 1, -a.getFloat(idx2 + 1));

            }
        }

        // ----------------------------------------------------------
        for (long s = 1; s < n1d2; s++) {
            idx1 = s * twoSliceStride;
            idx2 = (slicesl - s) * twoSliceStride;
            idx3 = n2d2 * twoRowStride;
            idx4 = idx1 + idx3;
            idx5 = idx2 + idx3;
            a.setFloat(idx1 + columnsl, a.getFloat(idx2 + 1));
            a.setFloat(idx2 + columnsl, a.getFloat(idx2 + 1));
            a.setFloat(idx1 + columnsl + 1, -a.getFloat(idx2));
            a.setFloat(idx2 + columnsl + 1, a.getFloat(idx2));
            a.setFloat(idx4 + columnsl, a.getFloat(idx5 + 1));
            a.setFloat(idx5 + columnsl, a.getFloat(idx5 + 1));
            a.setFloat(idx4 + columnsl + 1, -a.getFloat(idx5));
            a.setFloat(idx5 + columnsl + 1, a.getFloat(idx5));
            a.setFloat(idx2, a.getFloat(idx1));
            a.setFloat(idx2 + 1, -a.getFloat(idx1 + 1));
            a.setFloat(idx5, a.getFloat(idx4));
            a.setFloat(idx5 + 1, -a.getFloat(idx4 + 1));

        }

        // ----------------------------------------
        a.setFloat(columnsl, a.getFloat(1));
        a.setFloat(1, 0);
        idx1 = n2d2 * twoRowStride;
        idx2 = n1d2 * twoSliceStride;
        idx3 = idx1 + idx2;
        a.setFloat(idx1 + columnsl, a.getFloat(idx1 + 1));
        a.setFloat(idx1 + 1, 0);
        a.setFloat(idx2 + columnsl, a.getFloat(idx2 + 1));
        a.setFloat(idx2 + 1, 0);
        a.setFloat(idx3 + columnsl, a.getFloat(idx3 + 1));
        a.setFloat(idx3 + 1, 0);
        a.setFloat(idx2 + columnsl + 1, 0);
        a.setFloat(idx3 + columnsl + 1, 0);
    }

    /**
     * A test of {@link FloatFFT_3D#realForward(FloatLargeArray)}.
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
        if (!CommonUtils.isPowerOf2(numSlices)) {
            return;
        }
        int index;
        final FloatLargeArray actual = new FloatLargeArray(numSlices * numRows * 2 * numCols);
        final FloatLargeArray expected = new FloatLargeArray(numSlices * numRows * 2 * numCols);
        for (int s = 0; s < numSlices; s++) {
            for (int r = 0; r < numRows; r++) {
                for (int c = 0; c < numCols; c++) {
                    index = c + numCols * (r + numRows * s);
                    final float rnd = random.nextFloat();
                    actual.setFloat(index, rnd);
                    expected.setFloat(s * 2 * numRows * numCols + r * 2 * numCols + 2 * c, rnd);
                }
            }
        }
        fft.realForward(actual);
        fft.complexForward(expected);
        fillSymmetric(actual, numSlices, numRows, numCols);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numSlices, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    private void fillSymmetric(final float[][][] a, int slices, int rows, int columns)
    {
        final int twon3 = 2 * columns;
        final int n2d2 = rows / 2;
        int n1d2 = slices / 2;
        for (int s = 0; s < slices; s++) {
            int idx1 = (slices - s) % slices;
            for (int r = 0; r < rows; r++) {
                int idx2 = (rows - r) % rows;
                for (int c = 1; c < columns; c += 2) {
                    int idx3 = twon3 - c;
                    a[idx1][idx2][idx3] = -a[s][r][c + 2];
                    a[idx1][idx2][idx3 - 1] = a[s][r][c + 1];
                }
            }
        }

        // ---------------------------------------------
        for (int s = 0; s < slices; s++) {
            int idx1 = (slices - s) % slices;
            for (int r = 1; r < n2d2; r++) {
                int idx2 = rows - r;
                a[idx1][r][columns] = a[s][idx2][1];
                a[s][idx2][columns] = a[s][idx2][1];
                a[idx1][r][columns + 1] = -a[s][idx2][0];
                a[s][idx2][columns + 1] = a[s][idx2][0];
            }
        }

        for (int s = 0; s < slices; s++) {
            int idx1 = (slices - s) % slices;
            for (int r = 1; r < n2d2; r++) {
                int idx2 = rows - r;
                a[idx1][idx2][0] = a[s][r][0];
                a[idx1][idx2][1] = -a[s][r][1];
            }
        }
        // ----------------------------------------------------------
        for (int s = 1; s < n1d2; s++) {
            int idx1 = slices - s;
            a[s][0][columns] = a[idx1][0][1];
            a[idx1][0][columns] = a[idx1][0][1];
            a[s][0][columns + 1] = -a[idx1][0][0];
            a[idx1][0][columns + 1] = a[idx1][0][0];
            a[s][n2d2][columns] = a[idx1][n2d2][1];
            a[idx1][n2d2][columns] = a[idx1][n2d2][1];
            a[s][n2d2][columns + 1] = -a[idx1][n2d2][0];
            a[idx1][n2d2][columns + 1] = a[idx1][n2d2][0];
            a[idx1][0][0] = a[s][0][0];
            a[idx1][0][1] = -a[s][0][1];
            a[idx1][n2d2][0] = a[s][n2d2][0];
            a[idx1][n2d2][1] = -a[s][n2d2][1];

        }
        // ----------------------------------------

        a[0][0][columns] = a[0][0][1];
        a[0][0][1] = 0;
        a[0][n2d2][columns] = a[0][n2d2][1];
        a[0][n2d2][1] = 0;
        a[n1d2][0][columns] = a[n1d2][0][1];
        a[n1d2][0][1] = 0;
        a[n1d2][n2d2][columns] = a[n1d2][n2d2][1];
        a[n1d2][n2d2][1] = 0;
        a[n1d2][0][columns + 1] = 0;
        a[n1d2][n2d2][columns + 1] = 0;
    }

    /**
     * A test of {@link FloatFFT_3D#realForward(float[][][])}.
     */
    @Test
    public void testRealForward3dInput()
    {
        if (!CommonUtils.isPowerOf2(numRows)) {
            return;
        }
        if (!CommonUtils.isPowerOf2(numCols)) {
            return;
        }
        if (!CommonUtils.isPowerOf2(numSlices)) {
            return;
        }
        final float[][][] actual = new float[numSlices][numRows][2 * numCols];
        final float[][][] expected = new float[numSlices][numRows][2 * numCols];
        for (int s = 0; s < numSlices; s++) {
            for (int r = 0; r < numRows; r++) {
                for (int c = 0; c < numCols; c++) {
                    final float rnd = random.nextFloat();
                    actual[s][r][c] = rnd;
                    expected[s][r][2 * c] = rnd;
                }
            }
        }
        fft.realForward(actual);
        fft.complexForward(expected);
        fillSymmetric(actual, numSlices, numRows, numCols);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numSlices, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_3D#realInverse(float[], boolean)}, with the
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
        if (!CommonUtils.isPowerOf2(numSlices)) {
            return;
        }
        final float[] actual = new float[numRows * numCols * numSlices];
        final float[] expected = new float[actual.length];
        for (int i = 0; i < actual.length; i++) {
            final float rnd = random.nextFloat();
            actual[i] = rnd;
            expected[i] = rnd;
        }
        fft.realForward(actual);
        fft.realInverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numSlices, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_3D#realInverse(FloatLargeArray, boolean)}, with the
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
        if (!CommonUtils.isPowerOf2(numSlices)) {
            return;
        }
        final FloatLargeArray actual = new FloatLargeArray(numRows * numCols * numSlices);
        final FloatLargeArray expected = new FloatLargeArray(actual.length());
        for (int i = 0; i < actual.length(); i++) {
            final float rnd = random.nextFloat();
            actual.setFloat(i, rnd);
            expected.setFloat(i, rnd);
        }
        fft.realForward(actual);
        fft.realInverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numSlices, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_3D#realInverse(float[][][], boolean)}, with
     * the second parameter set to <code>true</code>.
     */
    @Test
    public void testRealInverseScaled3dInput()
    {
        if (!CommonUtils.isPowerOf2(numRows)) {
            return;
        }
        if (!CommonUtils.isPowerOf2(numCols)) {
            return;
        }
        if (!CommonUtils.isPowerOf2(numSlices)) {
            return;
        }
        final float[][][] actual = new float[numSlices][numRows][numCols];
        final float[][][] expected = new float[numSlices][numRows][numCols];
        for (int s = 0; s < numSlices; s++) {
            for (int r = 0; r < numRows; r++) {
                for (int c = 0; c < numCols; c++) {
                    final float rnd = random.nextFloat();
                    actual[s][r][c] = rnd;
                    expected[s][r][c] = rnd;
                }
            }
        }
        fft.realForward(actual);
        fft.realInverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numSlices, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_3D#realForwardFull(float[])}.
     */
    @Test
    public void testRealForwardFull1dInput()
    {
        final int n = numSlices * numRows * numCols;
        final float[] actual = new float[2 * n];
        final float[] expected = new float[2 * n];
        for (int index = 0; index < n; index++) {
            final float rnd = random.nextFloat();
            actual[index] = rnd;
            expected[2 * index] = rnd;
        }
        fft.complexForward(expected);
        fft.realForwardFull(actual);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numSlices, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_3D#realForwardFull(FloatLargeArray)}.
     */
    @Test
    public void testRealForwardFullLarge()
    {
        final int n = numSlices * numRows * numCols;
        final FloatLargeArray actual = new FloatLargeArray(2 * n);
        final FloatLargeArray expected = new FloatLargeArray(2 * n);
        for (int index = 0; index < n; index++) {
            final float rnd = random.nextFloat();
            actual.setFloat(index, rnd);
            expected.setFloat(2 * index, rnd);
        }
        fft.complexForward(expected);
        fft.realForwardFull(actual);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numSlices, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_3D#realForwardFull(float[][][]).
     */
    @Test
    public void testRealForwardFull3dInput()
    {
        final float[][][] actual = new float[numSlices][numRows][2 * numCols];
        final float[][][] expected = new float[numSlices][numRows][2 * numCols];
        for (int s = 0; s < numSlices; s++) {
            for (int r = 0; r < numRows; r++) {
                for (int c = 0; c < numCols; c++) {
                    final float rnd = random.nextFloat();
                    actual[s][r][c] = rnd;
                    expected[s][r][2 * c] = rnd;
                }
            }
        }
        fft.complexForward(expected);
        fft.realForwardFull(actual);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numSlices, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_3D#realInverseFull(float[], boolean)} with
     * the second parameter set to <code>true</code>.
     */
    @Test
    public void testRealInverseFullScaled1dInput()
    {
        final int n = numSlices * numRows * numCols;
        final float[] actual = new float[2 * n];
        final float[] expected = new float[2 * n];
        for (int index = 0; index < n; index++) {
            final float rnd = random.nextFloat();
            actual[index] = rnd;
            expected[2 * index] = rnd;
        }
        fft.complexInverse(expected, true);
        fft.realInverseFull(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numSlices, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_3D#realInverseFull(FloatLargeArray, boolean)} with
     * the second parameter set to <code>true</code>.
     */
    @Test
    public void testRealInverseFullScaledLarge()
    {
        final int n = numSlices * numRows * numCols;
        final FloatLargeArray actual = new FloatLargeArray(2 * n);
        final FloatLargeArray expected = new FloatLargeArray(2 * n);
        for (int index = 0; index < n; index++) {
            final float rnd = random.nextFloat();
            actual.setFloat(index, rnd);
            expected.setFloat(2 * index, rnd);
        }
        fft.complexInverse(expected, true);
        fft.realInverseFull(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numSlices, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * A test of {@link FloatFFT_3D#realInverseFull(float[][][], boolean)},
     * with the second parameter set to <code>true</code>.
     */
    @Test
    public void testRealInverseFullScaled3dInput()
    {
        final float[][][] actual = new float[numSlices][numRows][2 * numCols];
        final float[][][] expected = new float[numSlices][numRows][2 * numCols];
        for (int s = 0; s < numSlices; s++) {
            for (int r = 0; r < numRows; r++) {
                for (int c = 0; c < numCols; c++) {
                    final float rnd = random.nextFloat();
                    actual[s][r][c] = rnd;
                    expected[s][r][2 * c] = rnd;
                }
            }
        }
        fft.complexInverse(expected, true);
        fft.realInverseFull(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numSlices, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }
}
