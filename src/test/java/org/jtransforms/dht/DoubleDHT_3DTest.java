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
package org.jtransforms.dht;

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
import pl.edu.icm.jlargearrays.LargeArray;
import static org.apache.commons.math3.util.FastMath.*;

/**
 * This is a series of JUnit tests for the {@link DoubleDHT_3D}.
 *
 * @author Piotr Wendykier
 */
@RunWith(value = Parameterized.class)
public class DoubleDHT_3DTest
{

    /**
     * Base message of all exceptions.
     */
    public static final String DEFAULT_MESSAGE = "%d-threaded DHT of size %dx%dx%d: ";

    /**
     * The constant value of the seed of the random generator.
     */
    public static final int SEED = 20110602;

    private static final double EPS = pow(10, -12);

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
     * The DHT to be tested.
     */
    private final DoubleDHT_3D dht;

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
    public DoubleDHT_3DTest(final int numSlices, final int numRows,
                            final int numColumns, final int numThreads, final long seed)
    {
        this.numSlices = numSlices;
        this.numRows = numRows;
        this.numCols = numColumns;
        LargeArray.setMaxSizeOf32bitArray(1);
        this.dht = new DoubleDHT_3D(numSlices, numRows, numCols);
        this.random = new Random(seed);
        ConcurrencyUtils.setNumberOfThreads(numThreads);
        CommonUtils.setThreadsBeginN_3D(1);
        this.numThreads = ConcurrencyUtils.getNumberOfThreads();
    }

    /**
     * This is a test of {@link DoubleDHT_3D#forward(double[], boolean)},
     * and {@link DoubleDHT_3D#inverse(double[], boolean)}
     * with the second parameter set to <code>true</code>.
     */
    @Test
    public void testScaled()
    {
        final double[] actual = new double[numSlices * numRows * numCols];
        final double[] expected = new double[numSlices * numRows * numCols];
        for (int s = 0; s < numSlices; s++) {
            for (int r = 0; r < numRows; r++) {
                for (int c = 0; c < numCols; c++) {
                    final double rnd = random.nextDouble();
                    actual[s * numRows * numCols + r * numCols + c] = rnd;
                    expected[s * numRows * numCols + r * numCols + c] = rnd;
                }
            }
        }
        dht.forward(actual);
        dht.inverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numSlices, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link DoubleDHT_3D#forward(DoubleLargeArray, boolean)},
     * and {@link DoubleDHT_3D#inverse(DoubleLargeArray], boolean)}
     * with the second parameter set to <code>true</code>.
     */
    @Test
    public void testScaledLarge()
    {
        final DoubleLargeArray actual = new DoubleLargeArray(numSlices * numRows * numCols);
        final DoubleLargeArray expected = new DoubleLargeArray(numSlices * numRows * numCols);
        for (int s = 0; s < numSlices; s++) {
            for (int r = 0; r < numRows; r++) {
                for (int c = 0; c < numCols; c++) {
                    final double rnd = random.nextDouble();
                    actual.setDouble(s * numRows * numCols + r * numCols + c, rnd);
                    expected.setDouble(s * numRows * numCols + r * numCols + c, rnd);
                }
            }
        }
        dht.forward(actual);
        dht.inverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numSlices, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link DoubleDHT_3D#forward(double[][][], boolean)},
     * and {@link DoubleDHT_3D#inverse(double[][][], boolean)}
     * with the second parameter set to <code>true</code>.
     */
    @Test
    public void testScaled3D()
    {
        final double[][][] actual = new double[numSlices][numRows][numCols];
        final double[][][] expected = new double[numSlices][numRows][numCols];
        for (int s = 0; s < numSlices; s++) {
            for (int r = 0; r < numRows; r++) {
                for (int c = 0; c < numCols; c++) {
                    final double rnd = random.nextDouble();
                    actual[s][r][c] = rnd;
                    expected[s][r][c] = rnd;
                }
            }
        }
        dht.forward(actual);
        dht.inverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numSlices, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }
}
