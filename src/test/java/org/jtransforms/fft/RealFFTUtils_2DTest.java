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
import pl.edu.icm.jlargearrays.ConcurrencyUtils;
import org.jtransforms.utils.IOUtils;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import pl.edu.icm.jlargearrays.DoubleLargeArray;
import pl.edu.icm.jlargearrays.FloatLargeArray;
import static org.apache.commons.math3.util.FastMath.*;

/**
 * Test of the utility class {@link RealFFTUtils_2D}.
 *
 * @author S&eacute;bastien Brisard
 */
@RunWith(value = Parameterized.class)
public class RealFFTUtils_2DTest
{

    /**
     * Base message of all exceptions.
     */
    public static final String DEFAULT_MESSAGE = "%d-threaded FFT of size %dx%d: ";

    /**
     * The constant value of the seed of the random generator.
     */
    public static final int SEED = 20110624;

    private static final double EPSD = pow(10, -12);

    private static final double EPSF = pow(10, -3);

    @Parameters
    public static Collection<Object[]> getParameters()
    {
        final int[] size = {16, 32, 64, 128};

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
     * Number of columns of the data arrays to be Fourier transformed.
     */
    private final int columns;

    /**
     * To perform FFTs on double precision data.
     */
    private final DoubleFFT_2D fft2d;

    /**
     * To perform FFTs on single precision data.
     */
    private final FloatFFT_2D fft2f;

    /**
     * For the generation of the data arrays.
     */
    private final Random random;

    /**
     * Number of rows of the data arrays to be Fourier transformed.
     */
    private final int rows;

    /**
     * The object to be tested.
     */
    private final RealFFTUtils_2D unpacker;

    /**
     * The number of threads used.
     */
    private final int numThreads;

    /**
     * Creates a new instance of this test.
     *
     * @param rows
     *                   number of rows
     * @param columns
     *                   number of columns
     * @param numThreads
     *                   the number of threads to be used
     * @param seed
     *                   the seed of the random generator
     */
    public RealFFTUtils_2DTest(final int rows, final int columns,
                               final int numThreads, final long seed)
    {
        this.rows = rows;
        this.columns = columns;
        this.fft2d = new DoubleFFT_2D(rows, columns);
        this.fft2f = new FloatFFT_2D(rows, columns);
        this.random = new Random(seed);
        this.unpacker = new RealFFTUtils_2D(rows, columns);
        ConcurrencyUtils.setNumberOfThreads(numThreads);
        this.numThreads = ConcurrencyUtils.getNumberOfThreads();
    }

    @Test
    public void testUnpack1dInput()
    {

        final double[] actual0 = new double[rows * columns];
        final double[][] actual = new double[rows][2 * columns];
        final double[][] expected = new double[rows][2 * columns];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                final double rnd = random.nextDouble();
                actual0[r * columns + c] = rnd;
                expected[r][2 * c] = rnd;
            }
        }
        fft2d.complexForward(expected);
        fft2d.realForward(actual0);

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < 2 * columns; c++) {
                actual[r][c] = unpacker.unpack(r, c, actual0, 0);
            }
        }
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, rows, columns) + ", rmse = " + rmse, 0.0, rmse, EPSD);
    }

    @Test
    public void testUnpack1dInputLarge()
    {

        final DoubleLargeArray actual0 = new DoubleLargeArray(rows * columns);
        final double[][] actual = new double[rows][2 * columns];
        final double[][] expected = new double[rows][2 * columns];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                final double rnd = random.nextDouble();
                actual0.setDouble(r * columns + c, rnd);
                expected[r][2 * c] = rnd;
            }
        }
        fft2d.complexForward(expected);
        fft2d.realForward(actual0);

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < 2 * columns; c++) {
                actual[r][c] = unpacker.unpack(r, c, actual0, 0);
            }
        }
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, rows, columns) + ", rmse = " + rmse, 0.0, rmse, EPSD);
    }

    @Test
    public void testUnpack1fInput()
    {
        final float[] actual0 = new float[rows * columns];
        final float[][] actual = new float[rows][2 * columns];
        final float[][] expected = new float[rows][2 * columns];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                final float rnd = random.nextFloat();
                actual0[r * columns + c] = rnd;
                expected[r][2 * c] = rnd;
            }
        }
        fft2f.complexForward(expected);
        fft2f.realForward(actual0);

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < 2 * columns; c++) {
                actual[r][c] = unpacker.unpack(r, c, actual0, 0);
            }
        }
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, rows, columns) + ", rmse = " + rmse, 0.0, rmse, EPSF);
    }

    @Test
    public void testUnpack1fInputLarge()
    {
        final FloatLargeArray actual0 = new FloatLargeArray(rows * columns);
        final float[][] actual = new float[rows][2 * columns];
        final float[][] expected = new float[rows][2 * columns];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                final float rnd = random.nextFloat();
                actual0.setFloat(r * columns + c, rnd);
                expected[r][2 * c] = rnd;
            }
        }
        fft2f.complexForward(expected);
        fft2f.realForward(actual0);

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < 2 * columns; c++) {
                actual[r][c] = unpacker.unpack(r, c, actual0, 0);
            }
        }
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, rows, columns) + ", rmse = " + rmse, 0.0, rmse, EPSF);
    }

    @Test
    public void testUnpack2dInput()
    {
        final double[][] actual0 = new double[rows][columns];
        final double[][] actual = new double[rows][2 * columns];
        final double[][] expected = new double[rows][2 * columns];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                final double rnd = random.nextDouble();
                actual0[r][c] = rnd;
                expected[r][2 * c] = rnd;
            }
        }
        fft2d.complexForward(expected);
        fft2d.realForward(actual0);

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < 2 * columns; c++) {
                actual[r][c] = unpacker.unpack(r, c, actual0);
            }
        }
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, rows, columns) + ", rmse = " + rmse, 0.0, rmse, EPSD);
    }

    @Test
    public void testUnpack2fInput()
    {
        final float[][] actual0 = new float[rows][columns];
        final float[][] actual = new float[rows][2 * columns];
        final float[][] expected = new float[rows][2 * columns];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                final float rnd = random.nextFloat();
                actual0[r][c] = rnd;
                expected[r][2 * c] = rnd;
            }
        }
        fft2f.complexForward(expected);
        fft2f.realForward(actual0);

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < 2 * columns; c++) {
                actual[r][c] = unpacker.unpack(r, c, actual0);
            }
        }
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, rows, columns) + ", rmse = " + rmse, 0.0, rmse, EPSF);
    }

    @Test
    public void testPack1dInput()
    {
        final double[] data = new double[rows * columns];
        String msg = String.format(DEFAULT_MESSAGE, numThreads, rows, columns) + "[%d][%d]";
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < 2 * columns; c++) {
                final double expected = random.nextDouble();
                try {
                    unpacker.pack(expected, r, c, data, 0);
                    final double actual = unpacker.unpack(r, c, data, 0);
                    Assert.assertEquals(String.format(msg, r, c), expected,
                                        actual, 0.);
                } catch (IllegalArgumentException e) {
                    // Do nothing
                }
            }
        }
    }

    @Test
    public void testPack1dInputLarge()
    {
        final DoubleLargeArray data = new DoubleLargeArray(rows * columns);
        String msg = String.format(DEFAULT_MESSAGE, numThreads, rows, columns) + "[%d][%d]";
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < 2 * columns; c++) {
                final double expected = random.nextDouble();
                try {
                    unpacker.pack(expected, r, c, data, 0);
                    final double actual = unpacker.unpack(r, c, data, 0);
                    Assert.assertEquals(String.format(msg, r, c), expected,
                                        actual, 0.);
                } catch (IllegalArgumentException e) {
                    // Do nothing
                }
            }
        }
    }

    @Test
    public void testPack1fInput()
    {
        final float[] data = new float[rows * columns];
        String msg = String.format(DEFAULT_MESSAGE, numThreads, rows, columns) + "[%d][%d]";
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < 2 * columns; c++) {
                final float expected = random.nextFloat();
                try {
                    unpacker.pack(expected, r, c, data, 0);
                    final float actual = unpacker.unpack(r, c, data, 0);
                    Assert.assertEquals(String.format(msg, r, c), expected,
                                        actual, 0.);
                } catch (IllegalArgumentException e) {
                    // Do nothing
                }
            }
        }
    }

    @Test
    public void testPack1fInputLarge()
    {
        final FloatLargeArray data = new FloatLargeArray(rows * columns);
        String msg = String.format(DEFAULT_MESSAGE, numThreads, rows, columns) + "[%d][%d]";
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < 2 * columns; c++) {
                final float expected = random.nextFloat();
                try {
                    unpacker.pack(expected, r, c, data, 0);
                    final float actual = unpacker.unpack(r, c, data, 0);
                    Assert.assertEquals(String.format(msg, r, c), expected,
                                        actual, 0.);
                } catch (IllegalArgumentException e) {
                    // Do nothing
                }
            }
        }
    }

    @Test
    public void testPack2dInput()
    {
        final double[][] data = new double[rows][columns];
        String msg = String.format(DEFAULT_MESSAGE, numThreads, rows, columns) + "[%d][%d]";
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < 2 * columns; c++) {
                final double expected = random.nextDouble();
                try {
                    unpacker.pack(expected, r, c, data);
                    final double actual = unpacker.unpack(r, c, data);
                    Assert.assertEquals(String.format(msg, r, c), expected,
                                        actual, 0.);
                } catch (IllegalArgumentException e) {
                    // Do nothing
                }
            }
        }
    }

    @Test
    public void testPack2fInput()
    {
        final float[][] data = new float[rows][columns];
        String msg = String.format(DEFAULT_MESSAGE, numThreads, rows, columns) + "[%d][%d]";
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < 2 * columns; c++) {
                final float expected = random.nextFloat();
                try {
                    unpacker.pack(expected, r, c, data);
                    final float actual = unpacker.unpack(r, c, data);
                    Assert.assertEquals(String.format(msg, r, c), expected,
                                        actual, 0.);
                } catch (IllegalArgumentException e) {
                    // Do nothing
                }
            }
        }
    }
}
