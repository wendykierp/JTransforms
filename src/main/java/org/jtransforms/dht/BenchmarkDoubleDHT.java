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

import java.util.Arrays;
import org.jtransforms.utils.CommonUtils;
import pl.edu.icm.jlargearrays.ConcurrencyUtils;
import org.jtransforms.utils.IOUtils;
import pl.edu.icm.jlargearrays.DoubleLargeArray;

/**
 * Benchmark of double precision DHT's
 *
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class BenchmarkDoubleDHT
{

    private static int nthread = 8;

    private static int niter = 200;

    private static int nsize = 16;

    private static int threadsBegin2D = 65636;

    private static int threadsBegin3D = 65636;

    private static boolean doWarmup = true;

    private static long[] sizes1D = new long[]{262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 10368, 27000, 75600, 165375, 362880, 1562500, 3211264, 6250000};

    private static long[] sizes2D = new long[]{256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 260, 520, 1050, 1458, 1960, 2916, 4116, 5832};

    private static long[] sizes3D = new long[]{16, 32, 64, 128, 256, 512, 1024, 2048, 5, 17, 30, 95, 180, 270, 324, 420};

    private static boolean doScaling = false;

    private BenchmarkDoubleDHT()
    {

    }

    public static void parseArguments(String[] args)
    {
        if (args.length > 0) {
            nthread = Integer.parseInt(args[0]);
            threadsBegin2D = Integer.parseInt(args[1]);
            threadsBegin3D = Integer.parseInt(args[2]);
            niter = Integer.parseInt(args[3]);
            doWarmup = Boolean.parseBoolean(args[4]);
            doScaling = Boolean.parseBoolean(args[5]);
            nsize = Integer.parseInt(args[6]);
            sizes1D = new long[nsize];
            sizes2D = new long[nsize];
            sizes3D = new long[nsize];
            for (int i = 0; i < nsize; i++) {
                sizes1D[i] = Integer.parseInt(args[7 + i]);
            }
            for (int i = 0; i < nsize; i++) {
                sizes2D[i] = Integer.parseInt(args[7 + nsize + i]);
            }
            for (int i = 0; i < nsize; i++) {
                sizes3D[i] = Integer.parseInt(args[7 + nsize + nsize + i]);
            }
        } else {
            System.out.println("Default settings are used.");
        }
        ConcurrencyUtils.setNumberOfThreads(nthread);
        CommonUtils.setThreadsBeginN_2D(threadsBegin2D);
        CommonUtils.setThreadsBeginN_3D(threadsBegin3D);
        System.out.println("nthred = " + nthread);
        System.out.println("threadsBegin2D = " + threadsBegin2D);
        System.out.println("threadsBegin3D = " + threadsBegin3D);
        System.out.println("niter = " + niter);
        System.out.println("doWarmup = " + doWarmup);
        System.out.println("doScaling = " + doScaling);
        System.out.println("nsize = " + nsize);
        System.out.println("sizes1D[] = " + Arrays.toString(sizes1D));
        System.out.println("sizes2D[] = " + Arrays.toString(sizes2D));
        System.out.println("sizes3D[] = " + Arrays.toString(sizes3D));
    }

    public static void benchmarkForward_1D()
    {
        double[] times_without_constructor = new double[nsize];
        double[] times_with_constructor = new double[nsize];
        double[] x;
        for (int i = 0; i < nsize; i++) {
            System.out.println("Forward DHT 1D of size " + sizes1D[i]);
            if (doWarmup) { // call the transform twice to warm up
                DoubleDHT_1D dht = new DoubleDHT_1D(sizes1D[i]);
                x = new double[(int) sizes1D[i]];
                IOUtils.fillMatrix_1D(sizes1D[i], x);
                dht.forward(x);
                IOUtils.fillMatrix_1D(sizes1D[i], x);
                dht.forward(x);
            }
            long elapsedTime = System.nanoTime();
            DoubleDHT_1D dht = new DoubleDHT_1D(sizes1D[i]);
            times_with_constructor[i] = (System.nanoTime() - elapsedTime) / 1000000.0;
            x = new double[(int) sizes1D[i]];
            double min_time = Double.MAX_VALUE;
            for (int j = 0; j < niter; j++) {
                IOUtils.fillMatrix_1D(sizes1D[i], x);
                elapsedTime = System.nanoTime();
                dht.forward(x);
                elapsedTime = System.nanoTime() - elapsedTime;
                if (elapsedTime < min_time) {
                    min_time = elapsedTime;
                }
            }
            times_without_constructor[i] = (double) min_time / 1000000.0;
            times_with_constructor[i] += times_without_constructor[i];
            System.out.println("\tBest execution time without constructor: " + String.format("%.2f", times_without_constructor[i]) + " msec");
            System.out.println("\tBest execution time with constructor: " + String.format("%.2f", times_with_constructor[i]) + " msec");
            x = null;
            dht = null;
            System.gc();
            CommonUtils.sleep(5000);
        }
        IOUtils.writeFFTBenchmarkResultsToFile("benchmarkDoubleForwardDHT_1D.txt", nthread, niter, doWarmup, doScaling, sizes1D, times_without_constructor, times_with_constructor);
    }

    public static void benchmarkForward_2D_input_1D()
    {
        double[] times_without_constructor = new double[nsize];
        double[] times_with_constructor = new double[nsize];
        DoubleLargeArray x;
        for (int i = 0; i < nsize; i++) {
            System.out.println("Forward DHT 2D (input 1D) of size " + sizes2D[i] + " x " + sizes2D[i]);
            if (doWarmup) { // call the transform twice to warm up
                DoubleDHT_2D dht2 = new DoubleDHT_2D(sizes2D[i], sizes2D[i]);
                x = new DoubleLargeArray(sizes2D[i] * sizes2D[i], false);
                IOUtils.fillMatrix_2D(sizes2D[i], sizes2D[i], x);
                dht2.forward(x);
                IOUtils.fillMatrix_2D(sizes2D[i], sizes2D[i], x);
                dht2.forward(x);
            }
            long elapsedTime = System.nanoTime();
            DoubleDHT_2D dht2 = new DoubleDHT_2D(sizes2D[i], sizes2D[i]);
            times_with_constructor[i] = (System.nanoTime() - elapsedTime) / 1000000.0;
            x = new DoubleLargeArray(sizes2D[i] * sizes2D[i], false);
            double min_time = Double.MAX_VALUE;
            for (int j = 0; j < niter; j++) {
                IOUtils.fillMatrix_2D(sizes2D[i], sizes2D[i], x);
                elapsedTime = System.nanoTime();
                dht2.forward(x);
                elapsedTime = System.nanoTime() - elapsedTime;
                if (elapsedTime < min_time) {
                    min_time = elapsedTime;
                }
            }
            times_without_constructor[i] = (double) min_time / 1000000.0;
            times_with_constructor[i] += times_without_constructor[i];
            System.out.println("\tBest execution time without constructor: " + String.format("%.2f", times_without_constructor[i]) + " msec");
            System.out.println("\tBest execution time with constructor: " + String.format("%.2f", times_with_constructor[i]) + " msec");
            x = null;
            dht2 = null;
            System.gc();
            CommonUtils.sleep(5000);
        }
        IOUtils.writeFFTBenchmarkResultsToFile("benchmarkDoubleForwardDHT_2D_input_1D.txt", nthread, niter, doWarmup, doScaling, sizes2D, times_without_constructor, times_with_constructor);

    }

    public static void benchmarkForward_2D_input_2D()
    {
        double[] times_without_constructor = new double[nsize];
        double[] times_with_constructor = new double[nsize];
        double[][] x;
        for (int i = 0; i < nsize; i++) {
            System.out.println("Forward DHT 2D (input 2D) of size " + sizes2D[i] + " x " + sizes2D[i]);
            if (doWarmup) { // call the transform twice to warm up
                DoubleDHT_2D dht2 = new DoubleDHT_2D(sizes2D[i], sizes2D[i]);
                x = new double[(int) sizes2D[i]][(int) sizes2D[i]];
                IOUtils.fillMatrix_2D(sizes2D[i], sizes2D[i], x);
                dht2.forward(x);
                IOUtils.fillMatrix_2D(sizes2D[i], sizes2D[i], x);
                dht2.forward(x);
            }
            long elapsedTime = System.nanoTime();
            DoubleDHT_2D dht2 = new DoubleDHT_2D(sizes2D[i], sizes2D[i]);
            times_with_constructor[i] = (System.nanoTime() - elapsedTime) / 1000000.0;
            x = new double[(int) sizes2D[i]][(int) sizes2D[i]];
            double min_time = Double.MAX_VALUE;
            for (int j = 0; j < niter; j++) {
                IOUtils.fillMatrix_2D(sizes2D[i], sizes2D[i], x);
                elapsedTime = System.nanoTime();
                dht2.forward(x);
                elapsedTime = System.nanoTime() - elapsedTime;
                if (elapsedTime < min_time) {
                    min_time = elapsedTime;
                }
            }
            times_without_constructor[i] = (double) min_time / 1000000.0;
            times_with_constructor[i] += times_without_constructor[i];
            System.out.println("\tBest execution time without constructor: " + String.format("%.2f", times_without_constructor[i]) + " msec");
            System.out.println("\tBest execution time with constructor: " + String.format("%.2f", times_with_constructor[i]) + " msec");
            x = null;
            dht2 = null;
            System.gc();
            CommonUtils.sleep(5000);
        }
        IOUtils.writeFFTBenchmarkResultsToFile("benchmarkDoubleForwardDHT_2D_input_2D.txt", nthread, niter, doWarmup, doScaling, sizes2D, times_without_constructor, times_with_constructor);

    }

    public static void benchmarkForward_3D_input_1D()
    {
        double[] times_without_constructor = new double[nsize];
        double[] times_with_constructor = new double[nsize];
        DoubleLargeArray x;
        for (int i = 0; i < nsize; i++) {
            System.out.println("Forward DHT 3D (input 1D) of size " + sizes3D[i] + " x " + sizes3D[i] + " x " + sizes3D[i]);
            if (doWarmup) { // call the transform twice to warm up
                DoubleDHT_3D dht3 = new DoubleDHT_3D(sizes3D[i], sizes3D[i], sizes3D[i]);
                x = new DoubleLargeArray(sizes3D[i] * sizes3D[i] * sizes3D[i], false);
                IOUtils.fillMatrix_3D(sizes3D[i], sizes3D[i], sizes3D[i], x);
                dht3.forward(x);
                IOUtils.fillMatrix_3D(sizes3D[i], sizes3D[i], sizes3D[i], x);
                dht3.forward(x);
            }
            long elapsedTime = System.nanoTime();
            DoubleDHT_3D dht3 = new DoubleDHT_3D(sizes3D[i], sizes3D[i], sizes3D[i]);
            times_with_constructor[i] = (System.nanoTime() - elapsedTime) / 1000000.0;
            x = new DoubleLargeArray(sizes3D[i] * sizes3D[i] * sizes3D[i], false);
            double min_time = Double.MAX_VALUE;
            for (int j = 0; j < niter; j++) {
                IOUtils.fillMatrix_3D(sizes3D[i], sizes3D[i], sizes3D[i], x);
                elapsedTime = System.nanoTime();
                dht3.forward(x);
                elapsedTime = System.nanoTime() - elapsedTime;
                if (elapsedTime < min_time) {
                    min_time = elapsedTime;
                }
            }
            times_without_constructor[i] = (double) min_time / 1000000.0;
            times_with_constructor[i] += times_without_constructor[i];
            System.out.println("\tBest execution time without constructor: " + String.format("%.2f", times_without_constructor[i]) + " msec");
            System.out.println("\tBest execution time with constructor: " + String.format("%.2f", times_with_constructor[i]) + " msec");
            x = null;
            dht3 = null;
            System.gc();
            CommonUtils.sleep(5000);
        }
        IOUtils.writeFFTBenchmarkResultsToFile("benchmarkDoubleForwardDHT_3D_input_1D.txt", nthread, niter, doWarmup, doScaling, sizes3D, times_without_constructor, times_with_constructor);

    }

    public static void benchmarkForward_3D_input_3D()
    {
        double[] times_without_constructor = new double[nsize];
        double[] times_with_constructor = new double[nsize];
        double[][][] x;
        for (int i = 0; i < nsize; i++) {
            System.out.println("Forward DHT 3D (input 3D) of size " + sizes3D[i] + " x " + sizes3D[i] + " x " + sizes3D[i]);
            if (doWarmup) { // call the transform twice to warm up
                DoubleDHT_3D dht3 = new DoubleDHT_3D(sizes3D[i], sizes3D[i], sizes3D[i]);
                x = new double[(int) sizes3D[i]][(int) sizes3D[i]][(int) sizes3D[i]];
                IOUtils.fillMatrix_3D(sizes3D[i], sizes3D[i], sizes3D[i], x);
                dht3.forward(x);
                IOUtils.fillMatrix_3D(sizes3D[i], sizes3D[i], sizes3D[i], x);
                dht3.forward(x);
            }
            long elapsedTime = System.nanoTime();
            DoubleDHT_3D dht3 = new DoubleDHT_3D(sizes3D[i], sizes3D[i], sizes3D[i]);
            times_with_constructor[i] = (System.nanoTime() - elapsedTime) / 1000000.0;
            x = new double[(int) sizes3D[i]][(int) sizes3D[i]][(int) sizes3D[i]];
            double min_time = Double.MAX_VALUE;
            for (int j = 0; j < niter; j++) {
                IOUtils.fillMatrix_3D(sizes3D[i], sizes3D[i], sizes3D[i], x);
                elapsedTime = System.nanoTime();
                dht3.forward(x);
                elapsedTime = System.nanoTime() - elapsedTime;
                if (elapsedTime < min_time) {
                    min_time = elapsedTime;
                }
            }
            times_without_constructor[i] = (double) min_time / 1000000.0;
            times_with_constructor[i] += times_without_constructor[i];
            System.out.println("\tBest execution time without constructor: " + String.format("%.2f", times_without_constructor[i]) + " msec");
            System.out.println("\tBest execution time with constructor: " + String.format("%.2f", times_with_constructor[i]) + " msec");
            x = null;
            dht3 = null;
            System.gc();
            CommonUtils.sleep(5000);
        }
        IOUtils.writeFFTBenchmarkResultsToFile("benchmarkDoubleForwardDHT_3D_input_3D.txt", nthread, niter, doWarmup, doScaling, sizes3D, times_without_constructor, times_with_constructor);
    }

    public static void main(String[] args)
    {
        parseArguments(args);
        benchmarkForward_1D();
        benchmarkForward_2D_input_1D();
        benchmarkForward_2D_input_2D();
        benchmarkForward_3D_input_1D();
        benchmarkForward_3D_input_3D();
        System.exit(0);

    }
}
