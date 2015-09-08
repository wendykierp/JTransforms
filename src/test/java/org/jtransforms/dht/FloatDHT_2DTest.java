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
import pl.edu.icm.jlargearrays.FloatLargeArray;
import pl.edu.icm.jlargearrays.LargeArray;
import static org.apache.commons.math3.util.FastMath.*;

/**
 * This is a series of JUnit tests for the {@link FloatDHT_2D}.
 *
 * @author Piotr Wendykier
 */
@RunWith(value = Parameterized.class)
public class FloatDHT_2DTest
{

    /**
     * Base message of all exceptions.
     */
    public static final String DEFAULT_MESSAGE = "%d-threaded DHT of size %dx%d: ";

    /**
     * The constant value of the seed of the random generator.
     */
    public static final int SEED = 20110602;

    private static final double EPS = pow(10, -5);

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
     * The DHT to be tested.
     */
    private final FloatDHT_2D dht;

    /**
     * Number of columns of the data arrays to be Fourier transformed.
     */
    private final int numCols;

    /**
     * Number of rows of the data arrays to be Fourier transformed.
     */
    private final int numRows;
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
    public FloatDHT_2DTest(final int numRows, final int numColumns,
                           final int numThreads, final long seed)
    {
        this.numRows = numRows;
        this.numCols = numColumns;
        LargeArray.setMaxSizeOf32bitArray(1);
        this.dht = new FloatDHT_2D(numRows, numCols);
        this.random = new Random(seed);
        ConcurrencyUtils.setNumberOfThreads(numThreads);
        CommonUtils.setThreadsBeginN_2D(4096);
        this.numThreads = ConcurrencyUtils.getNumberOfThreads();
    }

    /**
     * This is a test of {@link FloatDHT_2D#forward(float[], boolean)},
     * and {@link FloatDHT_2D#inverse(float[], boolean)}
     * with the second parameter set to <code>true</code>.
     */
    @Test
    public void testScaled()
    {
        final float[] actual = new float[numRows * numCols];
        final float[] expected = new float[numRows * numCols];
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                final float rnd = random.nextFloat();
                actual[r * numCols + c] = rnd;
                expected[r * numCols + c] = rnd;
            }
        }
        dht.forward(actual);
        dht.inverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link FloatDHT_2D#forward(FloatLargeArray, boolean)},
     * and {@link FloatDHT_2D#inverse(FloatLargeArray], boolean)}
     * with the second parameter set to <code>true</code>.
     */
    @Test
    public void testScaledLarge()
    {
        final FloatLargeArray actual = new FloatLargeArray(numRows * numCols);
        final FloatLargeArray expected = new FloatLargeArray(numRows * numCols);
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                final float rnd = random.nextFloat();
                actual.setFloat(r * numCols + c, rnd);
                expected.setFloat(r * numCols + c, rnd);
            }
        }
        dht.forward(actual);
        dht.inverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of {@link FloatDHT_2D#forward(double[][], boolean)},
     * and {@link FloatDHT_2D#inverse(double[][], boolean)}
     * with the second parameter set to <code>true</code>.
     */
    @Test
    public void testScaled2D()
    {
        final float[][] actual = new float[numRows][numCols];
        final float[][] expected = new float[numRows][numCols];
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                final float rnd = random.nextFloat();
                actual[r][c] = rnd;
                expected[r][c] = rnd;
            }
        }
        dht.forward(actual);
        dht.inverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, numRows, numCols) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }
}
