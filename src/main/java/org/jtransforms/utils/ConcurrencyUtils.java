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
package org.jtransforms.utils;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;

/**
 * Concurrency utilities.
 *
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class ConcurrencyUtils {

    /**
     * Thread pool.
     */
    private static final ExecutorService THREAD_POOL = Executors.newCachedThreadPool(new CustomThreadFactory(new CustomExceptionHandler()));

    private static long THREADS_BEGIN_N_1D_FFT_2THREADS = 8192;

    private static long THREADS_BEGIN_N_1D_FFT_4THREADS = 65536;

    private static long THREADS_BEGIN_N_2D = 65536;

    private static long LARGE_ARAYS_BEGIN_N = (1 << 28);

    private static long THREADS_BEGIN_N_3D = 65536;

    private static int NTHREADS = prevPow2(getNumberOfProcessors());

    private ConcurrencyUtils() {

    }

    private static class CustomExceptionHandler implements Thread.UncaughtExceptionHandler {

        public void uncaughtException(Thread t, Throwable e) {
            e.printStackTrace();
        }

    }

    private static class CustomThreadFactory implements ThreadFactory {

        private static final ThreadFactory defaultFactory = Executors.defaultThreadFactory();

        private final Thread.UncaughtExceptionHandler handler;

        CustomThreadFactory(Thread.UncaughtExceptionHandler handler) {
            this.handler = handler;
        }

        public Thread newThread(Runnable r) {
            Thread t = defaultFactory.newThread(r);
            t.setUncaughtExceptionHandler(handler);
            return t;
        }
    };

    /**
     * Returns the number of available processors.
     *
     * @return number of available processors
     */
    public static int getNumberOfProcessors() {
        return Runtime.getRuntime().availableProcessors();
    }

    /**
     * Returns the current number of threads.
     *
     * @return the current number of threads.
     */
    public static int getNumberOfThreads() {
        return NTHREADS;
    }

    /**
     * Sets the number of threads. If n is not a power-of-two number, then the
     * number of threads is set to the closest power-of-two number less than n.
     *
     * @param n number of threads
     */
    public static void setNumberOfThreads(int n) {
        NTHREADS = prevPow2(n);
    }

    /**
     * Returns the minimal size of 1D data for which two threads are used.
     *
     * @return the minimal size of 1D data for which two threads are used
     */
    public static long getThreadsBeginN_1D_FFT_2Threads() {
        return THREADS_BEGIN_N_1D_FFT_2THREADS;
    }

    /**
     * Returns the minimal size of 1D data for which four threads are used.
     *
     * @return the minimal size of 1D data for which four threads are used
     */
    public static long getThreadsBeginN_1D_FFT_4Threads() {
        return THREADS_BEGIN_N_1D_FFT_4THREADS;
    }

    /**
     * Returns the minimal size of 2D data for which threads are used.
     *
     * @return the minimal size of 2D data for which threads are used
     */
    public static long getThreadsBeginN_2D() {
        return THREADS_BEGIN_N_2D;
    }

    /**
     * Returns the minimal size of 3D data for which threads are used.
     *
     * @return the minimal size of 3D data for which threads are used
     */
    public static long getThreadsBeginN_3D() {
        return THREADS_BEGIN_N_3D;
    }

    /**
     * Returns the minimal size for which JLargeArrays are used.
     *
     * @return the minimal size for which JLargeArrays are used
     */
    public static long getLargeArraysBeginN() {
        return LARGE_ARAYS_BEGIN_N;
    }

    /**
     * Sets the minimal size of 1D data for which two threads are used.
     *
     * @param n the minimal size of 1D data for which two threads are used
     */
    public static void setThreadsBeginN_1D_FFT_2Threads(long n) {
        if (n < 1024) {
            THREADS_BEGIN_N_1D_FFT_2THREADS = 1024;
        } else {
            THREADS_BEGIN_N_1D_FFT_2THREADS = n;
        }
    }

    /**
     * Sets the minimal size of 1D data for which four threads are used.
     *
     * @param n the minimal size of 1D data for which four threads are used
     */
    public static void setThreadsBeginN_1D_FFT_4Threads(long n) {
        if (n < 1024) {
            THREADS_BEGIN_N_1D_FFT_4THREADS = 1024;
        } else {
            THREADS_BEGIN_N_1D_FFT_4THREADS = n;
        }
    }

    /**
     * Sets the minimal size of 2D data for which threads are used.
     *
     * @param n the minimal size of 2D data for which threads are used
     */
    public static void setThreadsBeginN_2D(long n) {
        if (n < 4096) {
            THREADS_BEGIN_N_2D = 4096;
        } else {
            THREADS_BEGIN_N_2D = n;
        }
    }

    /**
     * Sets the minimal size of 3D data for which threads are used.
     *
     * @param n the minimal size of 3D data for which threads are used
     */
    public static void setThreadsBeginN_3D(long n) {
        THREADS_BEGIN_N_3D = n;
    }

    /**
     * Resets the minimal size of 1D data for which two and four threads are
     * used.
     */
    public static void resetThreadsBeginN_FFT() {
        THREADS_BEGIN_N_1D_FFT_2THREADS = 8192;
        THREADS_BEGIN_N_1D_FFT_4THREADS = 65536;
    }

    /**
     * Resets the minimal size of 2D and 3D data for which threads are used.
     */
    public static void resetThreadsBeginN() {
        THREADS_BEGIN_N_2D = 65536;
        THREADS_BEGIN_N_3D = 65536;
    }

    /**
     * Sets the minimal size for which JLargeArrays are used.
     *
     * @param n the maximal size for which JLargeArrays are used
     */
    public static void setLargeArraysBeginN(long n) {
        if (n < 1) {
            LARGE_ARAYS_BEGIN_N = 1;
        } else if (n > (1 << 28)) {
            LARGE_ARAYS_BEGIN_N = (1 << 28);
        } else {
            LARGE_ARAYS_BEGIN_N = n;
        }
    }

    /**
     * Returns the closest power-of-two number greater than or equal to x.
     *
     * @param x input value
     * @return the closest power-of-two number greater than or equal to x
     */
    public static int nextPow2(int x) {
        if (x < 1) {
            throw new IllegalArgumentException("x must be greater or equal 1");
        }
        if ((x & (x - 1)) == 0) {
            return x; // x is already a power-of-two number 
        }
        x |= (x >>> 1);
        x |= (x >>> 2);
        x |= (x >>> 4);
        x |= (x >>> 8);
        x |= (x >>> 16);
        return x + 1;
    }

    /**
     * Returns the closest power-of-two number greater than or equal to x.
     *
     * @param x input value
     * @return the closest power-of-two number greater than or equal to x
     */
    public static long nextPow2(long x) {
        if (x < 1) {
            throw new IllegalArgumentException("x must be greater or equal 1");
        }
        if ((x & (x - 1l)) == 0) {
            return x; // x is already a power-of-two number 
        }
        x |= (x >>> 1l);
        x |= (x >>> 2l);
        x |= (x >>> 4l);
        x |= (x >>> 8l);
        x |= (x >>> 16l);
        x |= (x >>> 32l);
        return x + 1l;
    }

    /**
     * Returns the closest power-of-two number less than or equal to x.
     *
     * @param x input value
     * @return the closest power-of-two number less then or equal to x
     */
    public static int prevPow2(int x) {
        if (x < 1) {
            throw new IllegalArgumentException("x must be greater or equal 1");
        }
        return (int) Math.pow(2, Math.floor(Math.log(x) / Math.log(2)));
    }

    /**
     * Returns the closest power-of-two number less than or equal to x.
     *
     * @param x input value 
     * @return the closest power-of-two number less then or equal to x
     */
    public static long prevPow2(long x) {
        if (x < 1) {
            throw new IllegalArgumentException("x must be greater or equal 1");
        }
        return (long) Math.pow(2, Math.floor(Math.log(x) / Math.log(2)));
    }

    /**
     * Checks if x is a power-of-two number.
     *
     * @param x input value
     * @return true if x is a power-of-two number
     */
    public static boolean isPowerOf2(int x) {
        if (x <= 0) {
            return false;
        } else {
            return (x & (x - 1)) == 0;
        }
    }

    /**
     * Checks if x is a power-of-two number.
     *
     * @param x input value
     * @return true if x is a power-of-two number
     */
    public static boolean isPowerOf2(long x) {
        if (x <= 0) {
            return false;
        } else {
            return (x & (x - 1l)) == 0;
        }
    }

    /**
     * Causes the currently executing thread to sleep (temporarily cease
     * execution) for the specified number of milliseconds.
     *
     * @param millis the length of time to sleep in milliseconds
     */
    public static void sleep(long millis) {
        try {
            Thread.sleep(5000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    /**
     * Submits a Runnable task for execution and returns a Future representing
     * that task.
     *
     * @param task a Runnable task for execution
     * @return a Future representing the task
     */
    public static Future<?> submit(Runnable task) {
        return THREAD_POOL.submit(task);
    }

    /**
     * Waits for all threads to complete computation.
     *
     * @param futures array of Future objects 
     */
    public static void waitForCompletion(Future<?>[] futures) {
        int size = futures.length;
        try {
            for (int j = 0; j < size; j++) {
                futures[j].get();
            }
        } catch (ExecutionException ex) {
            ex.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
