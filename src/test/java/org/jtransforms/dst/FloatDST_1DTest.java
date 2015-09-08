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
package org.jtransforms.dst;

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
 * This is a series of JUnit tests for the {@link FloatDST_1D}.
 *
 * @author Piotr Wendykier
 */
@RunWith(value = Parameterized.class)
public class FloatDST_1DTest
{

    /**
     * Base message of all exceptions.
     */
    public static final String DEFAULT_MESSAGE = "%d-threaded DST of size %d: ";

    /**
     * The constant value of the seed of the random generator.
     */
    public static final int SEED = 20110602;

    private static final double EPS = pow(10, -5);

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
     * The DST to be tested.
     */
    private final FloatDST_1D dst;

    /**
     * The size of the DST to be tested.
     */
    private final int n;

    /**
     * For the generation of the data arrays.
     */
    private final Random random;

    /**
     * The number of threads used.
     */
    private final int numThreads;

    /**
     * Creates a new instance of this class.
     *
     * @param n
     *                   the size of the DST to be tested
     * @param numThreads
     *                   the number of threads
     * @param seed
     *                   the seed of the random generator
     */
    public FloatDST_1DTest(final int n, final int numThreads, final long seed)
    {
        this.n = n;
        LargeArray.setMaxSizeOf32bitArray(1);
        this.dst = new FloatDST_1D(n);
        this.random = new Random(seed);
        CommonUtils.setThreadsBeginN_1D_FFT_2Threads(512);
        CommonUtils.setThreadsBeginN_1D_FFT_4Threads(512);
        ConcurrencyUtils.setNumberOfThreads(numThreads);
        this.numThreads = ConcurrencyUtils.getNumberOfThreads();
    }

    /**
     * This is a test of {@link FloatDST_1D#forward(float[], boolean)},
     * and {@link FloatDST_1D#inverse(float[], boolean)}
     * with the second parameter set to <code>true</code>.
     */
    @Test
    public void testScaled()
    {
        final float[] actual = new float[2 * n];
        final float[] expected = new float[2 * n];
        for (int i = 0; i < 2 * n; i++) {
            actual[i] = 2.f * random.nextFloat() - 1.f;
            expected[i] = actual[i];
        }
        dst.forward(actual, true);
        dst.inverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }

    /**
     * This is a test of
     * {@link FloatDST_1D#forward(FloatLargeArray, boolean)}, and
     * {@link FloatDST_1D#inverse(FloatLargeArray, boolean)} with the second
     * parameter set to <code>true</code>.
     */
    @Test
    public void testScaledLarge()
    {
        final FloatLargeArray actual = new FloatLargeArray(n);
        final FloatLargeArray expected = new FloatLargeArray(n);
        for (int i = 0; i < n; i++) {
            actual.setFloat(i, 2.f * random.nextFloat() - 1.f);
            expected.setFloat(i, actual.getFloat(i));
        }
        dst.forward(actual, true);
        dst.inverse(actual, true);
        double rmse = IOUtils.computeRMSE(actual, expected);
        Assert.assertEquals(String.format(DEFAULT_MESSAGE, numThreads, n) + ", rmse = " + rmse, 0.0, rmse, EPS);
    }
}
