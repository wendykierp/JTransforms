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
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.logging.Logger;
import static org.apache.commons.math3.util.FastMath.*;
import pl.edu.icm.jlargearrays.ConcurrencyUtils;
import pl.edu.icm.jlargearrays.DoubleLargeArray;
import pl.edu.icm.jlargearrays.FloatLargeArray;
import pl.edu.icm.jlargearrays.LongLargeArray;

/**
 * Static utility methods.
 *  
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class CommonUtils
{

    private static long THREADS_BEGIN_N_1D_FFT_2THREADS = 8192;

    private static long THREADS_BEGIN_N_1D_FFT_4THREADS = 65536;

    private static long THREADS_BEGIN_N_2D = 65536;

    private static long THREADS_BEGIN_N_3D = 65536;

    private static boolean useLargeArrays = false;

    public CommonUtils()
    {
    }

    /**
     * Causes the currently executing thread to sleep (temporarily cease
     * execution) for the specified number of milliseconds.
     *
     * @param millis the length of time to sleep in milliseconds
     */
    public static void sleep(long millis)
    {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    /**
     * Returns the minimal size of 1D data for which two threads are used.
     *
     * @return the minimal size of 1D data for which two threads are used
     */
    public static long getThreadsBeginN_1D_FFT_2Threads()
    {
        return THREADS_BEGIN_N_1D_FFT_2THREADS;
    }

    /**
     * Returns the minimal size of 1D data for which four threads are used.
     *
     * @return the minimal size of 1D data for which four threads are used
     */
    public static long getThreadsBeginN_1D_FFT_4Threads()
    {
        return THREADS_BEGIN_N_1D_FFT_4THREADS;
    }

    /**
     * Returns the minimal size of 2D data for which threads are used.
     *
     * @return the minimal size of 2D data for which threads are used
     */
    public static long getThreadsBeginN_2D()
    {
        return THREADS_BEGIN_N_2D;
    }

    /**
     * Returns the minimal size of 3D data for which threads are used.
     *
     * @return the minimal size of 3D data for which threads are used
     */
    public static long getThreadsBeginN_3D()
    {
        return THREADS_BEGIN_N_3D;
    }

    /**
     * Sets the minimal size of 1D data for which two threads are used.
     *
     * @param n the minimal size of 1D data for which two threads are used
     */
    public static void setThreadsBeginN_1D_FFT_2Threads(long n)
    {
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
    public static void setThreadsBeginN_1D_FFT_4Threads(long n)
    {
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
    public static void setThreadsBeginN_2D(long n)
    {
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
    public static void setThreadsBeginN_3D(long n)
    {
        THREADS_BEGIN_N_3D = n;
    }

    /**
     * Resets the minimal size of 1D data for which two and four threads are
     * used.
     */
    public static void resetThreadsBeginN_FFT()
    {
        THREADS_BEGIN_N_1D_FFT_2THREADS = 8192;
        THREADS_BEGIN_N_1D_FFT_4THREADS = 65536;
    }

    /**
     * Resets the minimal size of 2D and 3D data for which threads are used.
     */
    public static void resetThreadsBeginN()
    {
        THREADS_BEGIN_N_2D = 65536;
        THREADS_BEGIN_N_3D = 65536;
    }

    /**
     * Returns the value of useLargeArrays variable.
     *  
     * @return the value of useLargeArrays variable
     */
    public static boolean isUseLargeArrays()
    {
        return useLargeArrays;
    }

    /**
     * Sets the value of useLargeArrays variable.
     *  
     * @param useLargeArrays the value of useLargeArrays variable
     */
    public static void setUseLargeArrays(boolean useLargeArrays)
    {
        CommonUtils.useLargeArrays = useLargeArrays;
    }

    /**
     * Returns the closest power-of-two number greater than or equal to x.
     *
     * @param x input value
     *
     * @return the closest power-of-two number greater than or equal to x
     */
    public static int nextPow2(int x)
    {
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
     *
     * @return the closest power-of-two number greater than or equal to x
     */
    public static long nextPow2(long x)
    {
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
     *
     * @return the closest power-of-two number less then or equal to x
     */
    public static int prevPow2(int x)
    {
        if (x < 1) {
            throw new IllegalArgumentException("x must be greater or equal 1");
        }
        return (int) pow(2, floor(log(x) / log(2)));
    }

    /**
     * Returns the closest power-of-two number less than or equal to x.
     *
     * @param x input value
     *
     * @return the closest power-of-two number less then or equal to x
     */
    public static long prevPow2(long x)
    {
        if (x < 1) {
            throw new IllegalArgumentException("x must be greater or equal 1");
        }
        return (long) pow(2, floor(log(x) / log(2)));
    }

    /**
     * Checks if x is a power-of-two number.
     *
     * @param x input value
     *
     * @return true if x is a power-of-two number
     */
    public static boolean isPowerOf2(int x)
    {
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
     *
     * @return true if x is a power-of-two number
     */
    public static boolean isPowerOf2(long x)
    {
        if (x <= 0) {
            return false;
        } else {
            return (x & (x - 1l)) == 0;
        }
    }

    public static long getReminder(long n, int factors[])
    {
        long reminder = n;

        if (n <= 0) {
            throw new IllegalArgumentException("n must be positive integer");
        }

        for (int i = 0; i < factors.length && reminder != 1l; i++) {
            long factor = factors[i];
            while ((reminder % factor) == 0) {
                reminder /= factor;
            }
        }
        return reminder;
    }

    public static void makeipt(int nw, int[] ip)
    {
        int j, l, m, m2, p, q;

        ip[2] = 0;
        ip[3] = 16;
        m = 2;
        for (l = nw; l > 32; l >>= 2) {
            m2 = m << 1;
            q = m2 << 3;
            for (j = m; j < m2; j++) {
                p = ip[j] << 2;
                ip[m + j] = p;
                ip[m2 + j] = p + q;
            }
            m = m2;
        }
    }

    public static void makeipt(long nw, LongLargeArray ipl)
    {
        long j, l, m, m2, p, q;

        ipl.setLong(2, 0);
        ipl.setLong(3, 16);
        m = 2;
        for (l = nw; l > 32; l >>= 2l) {
            m2 = m << 1l;
            q = m2 << 3l;
            for (j = m; j < m2; j++) {
                p = ipl.getLong(j) << 2l;
                ipl.setLong(m + j, p);
                ipl.setLong(m2 + j, p + q);
            }
            m = m2;
        }
    }

    public static void makewt(int nw, int[] ip, double[] w)
    {
        int j, nwh, nw0, nw1;
        double delta, wn4r, wk1r, wk1i, wk3r, wk3i;
        double delta2, deltaj, deltaj3;

        ip[0] = nw;
        ip[1] = 1;
        if (nw > 2) {
            nwh = nw >> 1;
            delta = 0.785398163397448278999490867136046290 / nwh;
            delta2 = delta * 2;
            wn4r = cos(delta * nwh);
            w[0] = 1;
            w[1] = wn4r;
            if (nwh == 4) {
                w[2] = cos(delta2);
                w[3] = sin(delta2);
            } else if (nwh > 4) {
                CommonUtils.makeipt(nw, ip);
                w[2] = 0.5 / cos(delta2);
                w[3] = 0.5 / cos(delta * 6);
                for (j = 4; j < nwh; j += 4) {
                    deltaj = delta * j;
                    deltaj3 = 3 * deltaj;
                    w[j] = cos(deltaj);
                    w[j + 1] = sin(deltaj);
                    w[j + 2] = cos(deltaj3);
                    w[j + 3] = -sin(deltaj3);
                }
            }
            nw0 = 0;
            while (nwh > 2) {
                nw1 = nw0 + nwh;
                nwh >>= 1;
                w[nw1] = 1;
                w[nw1 + 1] = wn4r;
                if (nwh == 4) {
                    wk1r = w[nw0 + 4];
                    wk1i = w[nw0 + 5];
                    w[nw1 + 2] = wk1r;
                    w[nw1 + 3] = wk1i;
                } else if (nwh > 4) {
                    wk1r = w[nw0 + 4];
                    wk3r = w[nw0 + 6];
                    w[nw1 + 2] = 0.5 / wk1r;
                    w[nw1 + 3] = 0.5 / wk3r;
                    for (j = 4; j < nwh; j += 4) {
                        int idx1 = nw0 + 2 * j;
                        int idx2 = nw1 + j;
                        wk1r = w[idx1];
                        wk1i = w[idx1 + 1];
                        wk3r = w[idx1 + 2];
                        wk3i = w[idx1 + 3];
                        w[idx2] = wk1r;
                        w[idx2 + 1] = wk1i;
                        w[idx2 + 2] = wk3r;
                        w[idx2 + 3] = wk3i;
                    }
                }
                nw0 = nw1;
            }
        }
    }

    public static void makewt(long nw, LongLargeArray ipl, DoubleLargeArray wl)
    {
        long j, nwh, nw0, nw1;
        double delta, wn4r, wk1r, wk1i, wk3r, wk3i;
        double delta2, deltaj, deltaj3;

        ipl.setLong(0, nw);
        ipl.setLong(1, 1l);
        if (nw > 2) {
            nwh = nw >> 1;
            delta = 0.785398163397448278999490867136046290 / nwh;
            delta2 = delta * 2;
            wn4r = cos(delta * nwh);
            wl.setDouble(0, 1);
            wl.setDouble(1, wn4r);
            if (nwh == 4) {
                wl.setDouble(2, cos(delta2));
                wl.setDouble(3, sin(delta2));
            } else if (nwh > 4) {
                CommonUtils.makeipt(nw, ipl);
                wl.setDouble(2, 0.5 / cos(delta2));
                wl.setDouble(3, 0.5 / cos(delta * 6));
                for (j = 4; j < nwh; j += 4) {
                    deltaj = delta * j;
                    deltaj3 = 3 * deltaj;
                    wl.setDouble(j, cos(deltaj));
                    wl.setDouble(j + 1, sin(deltaj));
                    wl.setDouble(j + 2, cos(deltaj3));
                    wl.setDouble(j + 3, -sin(deltaj3));
                }
            }
            nw0 = 0;
            while (nwh > 2) {
                nw1 = nw0 + nwh;
                nwh >>= 1l;
                wl.setDouble(nw1, 1);
                wl.setDouble(nw1 + 1, wn4r);
                if (nwh == 4) {
                    wk1r = wl.getDouble(nw0 + 4);
                    wk1i = wl.getDouble(nw0 + 5);
                    wl.setDouble(nw1 + 2, wk1r);
                    wl.setDouble(nw1 + 3, wk1i);
                } else if (nwh > 4) {
                    wk1r = wl.getDouble(nw0 + 4);
                    wk3r = wl.getDouble(nw0 + 6);
                    wl.setDouble(nw1 + 2, 0.5 / wk1r);
                    wl.setDouble(nw1 + 3, 0.5 / wk3r);
                    for (j = 4; j < nwh; j += 4) {
                        long idx1 = nw0 + 2 * j;
                        long idx2 = nw1 + j;
                        wk1r = wl.getDouble(idx1);
                        wk1i = wl.getDouble(idx1 + 1);
                        wk3r = wl.getDouble(idx1 + 2);
                        wk3i = wl.getDouble(idx1 + 3);
                        wl.setDouble(idx2, wk1r);
                        wl.setDouble(idx2 + 1, wk1i);
                        wl.setDouble(idx2 + 2, wk3r);
                        wl.setDouble(idx2 + 3, wk3i);
                    }
                }
                nw0 = nw1;
            }
        }
    }

    public static void makect(int nc, double[] c, int startc, int[] ip)
    {
        int j, nch;
        double delta, deltaj;

        ip[1] = nc;
        if (nc > 1) {
            nch = nc >> 1;
            delta = 0.785398163397448278999490867136046290 / nch;
            c[startc] = cos(delta * nch);
            c[startc + nch] = 0.5 * c[startc];
            for (j = 1; j < nch; j++) {
                deltaj = delta * j;
                c[startc + j] = 0.5 * cos(deltaj);
                c[startc + nc - j] = 0.5 * sin(deltaj);
            }
        }
    }

    public static void makect(long nc, DoubleLargeArray c, long startc, LongLargeArray ipl)
    {
        long j, nch;
        double delta, deltaj;

        ipl.setLong(1, nc);
        if (nc > 1) {
            nch = nc >> 1l;
            delta = 0.785398163397448278999490867136046290 / nch;
            c.setDouble(startc, cos(delta * nch));
            c.setDouble(startc + nch, 0.5 * c.getDouble(startc));
            for (j = 1; j < nch; j++) {
                deltaj = delta * j;
                c.setDouble(startc + j, 0.5 * cos(deltaj));
                c.setDouble(startc + nc - j, 0.5 * sin(deltaj));
            }
        }
    }

    public static void makect(int nc, float[] c, int startc, int[] ip)
    {
        int j, nch;
        float delta, deltaj;

        ip[1] = nc;
        if (nc > 1) {
            nch = nc >> 1;
            delta = 0.785398163397448278999490867136046290f / nch;
            c[startc] = (float) cos(delta * nch);
            c[startc + nch] = 0.5f * c[startc];
            for (j = 1; j < nch; j++) {
                deltaj = delta * j;
                c[startc + j] = 0.5f * (float) cos(deltaj);
                c[startc + nc - j] = 0.5f * (float) sin(deltaj);
            }
        }
    }

    public static void makect(long nc, FloatLargeArray c, long startc, LongLargeArray ipl)
    {
        long j, nch;
        float delta, deltaj;

        ipl.setLong(1, nc);
        if (nc > 1) {
            nch = nc >> 1l;
            delta = 0.785398163397448278999490867136046290f / nch;
            c.setFloat(startc, (float) cos(delta * nch));
            c.setFloat(startc + nch, 0.5f * c.getFloat(startc));
            for (j = 1; j < nch; j++) {
                deltaj = delta * j;
                c.setFloat(startc + j, 0.5f * (float) cos(deltaj));
                c.setFloat(startc + nc - j, 0.5f * (float) sin(deltaj));
            }
        }
    }

    public static void makewt(int nw, int[] ip, float[] w)
    {
        int j, nwh, nw0, nw1;
        float delta, wn4r, wk1r, wk1i, wk3r, wk3i;
        float delta2, deltaj, deltaj3;

        ip[0] = nw;
        ip[1] = 1;
        if (nw > 2) {
            nwh = nw >> 1;
            delta = 0.785398163397448278999490867136046290f / nwh;
            delta2 = delta * 2;
            wn4r = (float) cos(delta * nwh);
            w[0] = 1;
            w[1] = wn4r;
            if (nwh == 4) {
                w[2] = (float) cos(delta2);
                w[3] = (float) sin(delta2);
            } else if (nwh > 4) {
                CommonUtils.makeipt(nw, ip);
                w[2] = 0.5f / (float) cos(delta2);
                w[3] = 0.5f / (float) cos(delta * 6);
                for (j = 4; j < nwh; j += 4) {
                    deltaj = delta * j;
                    deltaj3 = 3 * deltaj;
                    w[j] = (float) cos(deltaj);
                    w[j + 1] = (float) sin(deltaj);
                    w[j + 2] = (float) cos(deltaj3);
                    w[j + 3] = -(float) sin(deltaj3);
                }
            }
            nw0 = 0;
            while (nwh > 2) {
                nw1 = nw0 + nwh;
                nwh >>= 1;
                w[nw1] = 1;
                w[nw1 + 1] = wn4r;
                if (nwh == 4) {
                    wk1r = w[nw0 + 4];
                    wk1i = w[nw0 + 5];
                    w[nw1 + 2] = wk1r;
                    w[nw1 + 3] = wk1i;
                } else if (nwh > 4) {
                    wk1r = w[nw0 + 4];
                    wk3r = w[nw0 + 6];
                    w[nw1 + 2] = 0.5f / wk1r;
                    w[nw1 + 3] = 0.5f / wk3r;
                    for (j = 4; j < nwh; j += 4) {
                        int idx1 = nw0 + 2 * j;
                        int idx2 = nw1 + j;
                        wk1r = w[idx1];
                        wk1i = w[idx1 + 1];
                        wk3r = w[idx1 + 2];
                        wk3i = w[idx1 + 3];
                        w[idx2] = wk1r;
                        w[idx2 + 1] = wk1i;
                        w[idx2 + 2] = wk3r;
                        w[idx2 + 3] = wk3i;
                    }
                }
                nw0 = nw1;
            }
        }
    }

    public static void makewt(long nw, LongLargeArray ipl, FloatLargeArray wl)
    {
        long j, nwh, nw0, nw1;
        float delta, wn4r, wk1r, wk1i, wk3r, wk3i;
        float delta2, deltaj, deltaj3;

        ipl.setLong(0, nw);
        ipl.setLong(1, 1l);
        if (nw > 2) {
            nwh = nw >> 1;
            delta = 0.785398163397448278999490867136046290f / nwh;
            delta2 = delta * 2;
            wn4r = (float) cos(delta * nwh);
            wl.setFloat(0, 1);
            wl.setFloat(1, wn4r);
            if (nwh == 4) {
                wl.setFloat(2, (float) cos(delta2));
                wl.setFloat(3, (float) sin(delta2));
            } else if (nwh > 4) {
                CommonUtils.makeipt(nw, ipl);
                wl.setFloat(2, 0.5f / (float) cos(delta2));
                wl.setFloat(3, 0.5f / (float) cos(delta * 6));
                for (j = 4; j < nwh; j += 4) {
                    deltaj = delta * j;
                    deltaj3 = 3 * deltaj;
                    wl.setFloat(j, (float) cos(deltaj));
                    wl.setFloat(j + 1, (float) sin(deltaj));
                    wl.setFloat(j + 2, (float) cos(deltaj3));
                    wl.setFloat(j + 3, -(float) sin(deltaj3));
                }
            }
            nw0 = 0;
            while (nwh > 2) {
                nw1 = nw0 + nwh;
                nwh >>= 1l;
                wl.setFloat(nw1, 1);
                wl.setFloat(nw1 + 1, wn4r);
                if (nwh == 4) {
                    wk1r = wl.getFloat(nw0 + 4);
                    wk1i = wl.getFloat(nw0 + 5);
                    wl.setFloat(nw1 + 2, wk1r);
                    wl.setFloat(nw1 + 3, wk1i);
                } else if (nwh > 4) {
                    wk1r = wl.getFloat(nw0 + 4);
                    wk3r = wl.getFloat(nw0 + 6);
                    wl.setFloat(nw1 + 2, 0.5f / wk1r);
                    wl.setFloat(nw1 + 3, 0.5f / wk3r);
                    for (j = 4; j < nwh; j += 4) {
                        long idx1 = nw0 + 2 * j;
                        long idx2 = nw1 + j;
                        wk1r = wl.getFloat(idx1);
                        wk1i = wl.getFloat(idx1 + 1);
                        wk3r = wl.getFloat(idx1 + 2);
                        wk3i = wl.getFloat(idx1 + 3);
                        wl.setFloat(idx2, wk1r);
                        wl.setFloat(idx2 + 1, wk1i);
                        wl.setFloat(idx2 + 2, wk3r);
                        wl.setFloat(idx2 + 3, wk3i);
                    }
                }
                nw0 = nw1;
            }
        }
    }

    public static void cftfsub(int n, double[] a, int offa, int[] ip, int nw, double[] w)
    {
        if (n > 8) {
            if (n > 32) {
                cftf1st(n, a, offa, w, nw - (n >> 2));
                if ((ConcurrencyUtils.getNumberOfThreads() > 1) && (n >= CommonUtils.getThreadsBeginN_1D_FFT_2Threads())) {
                    cftrec4_th(n, a, offa, nw, w);
                } else if (n > 512) {
                    cftrec4(n, a, offa, nw, w);
                } else if (n > 128) {
                    cftleaf(n, 1, a, offa, nw, w);
                } else {
                    cftfx41(n, a, offa, nw, w);
                }
                bitrv2(n, ip, a, offa);
            } else if (n == 32) {
                cftf161(a, offa, w, nw - 8);
                bitrv216(a, offa);
            } else {
                cftf081(a, offa, w, 0);
                bitrv208(a, offa);
            }
        } else if (n == 8) {
            cftf040(a, offa);
        } else if (n == 4) {
            cftxb020(a, offa);
        }
    }

    public static void cftfsub(long n, DoubleLargeArray a, long offa, LongLargeArray ip, long nw, DoubleLargeArray w)
    {
        if (n > 8) {
            if (n > 32) {
                cftf1st(n, a, offa, w, nw - (n >> 2l));
                if ((ConcurrencyUtils.getNumberOfThreads() > 1) && (n >= CommonUtils.getThreadsBeginN_1D_FFT_2Threads())) {
                    cftrec4_th(n, a, offa, nw, w);
                } else if (n > 512) {
                    cftrec4(n, a, offa, nw, w);
                } else if (n > 128) {
                    cftleaf(n, 1, a, offa, nw, w);
                } else {
                    cftfx41(n, a, offa, nw, w);
                }
                bitrv2l(n, ip, a, offa);
            } else if (n == 32) {
                cftf161(a, offa, w, nw - 8);
                bitrv216(a, offa);
            } else {
                cftf081(a, offa, w, 0);
                bitrv208(a, offa);
            }
        } else if (n == 8) {
            cftf040(a, offa);
        } else if (n == 4) {
            cftxb020(a, offa);
        }
    }

    public static void cftbsub(int n, double[] a, int offa, int[] ip, int nw, double[] w)
    {
        if (n > 8) {
            if (n > 32) {
                cftb1st(n, a, offa, w, nw - (n >> 2));
                if ((ConcurrencyUtils.getNumberOfThreads() > 1) && (n >= CommonUtils.getThreadsBeginN_1D_FFT_2Threads())) {
                    cftrec4_th(n, a, offa, nw, w);
                } else if (n > 512) {
                    cftrec4(n, a, offa, nw, w);
                } else if (n > 128) {
                    cftleaf(n, 1, a, offa, nw, w);
                } else {
                    cftfx41(n, a, offa, nw, w);
                }
                bitrv2conj(n, ip, a, offa);
            } else if (n == 32) {
                cftf161(a, offa, w, nw - 8);
                bitrv216neg(a, offa);
            } else {
                cftf081(a, offa, w, 0);
                bitrv208neg(a, offa);
            }
        } else if (n == 8) {
            cftb040(a, offa);
        } else if (n == 4) {
            cftxb020(a, offa);
        }
    }

    public static void cftbsub(long n, DoubleLargeArray a, long offa, LongLargeArray ip, long nw, DoubleLargeArray w)
    {
        if (n > 8) {
            if (n > 32) {
                cftb1st(n, a, offa, w, nw - (n >> 2l));
                if ((ConcurrencyUtils.getNumberOfThreads() > 1) && (n >= CommonUtils.getThreadsBeginN_1D_FFT_2Threads())) {
                    cftrec4_th(n, a, offa, nw, w);
                } else if (n > 512) {
                    cftrec4(n, a, offa, nw, w);
                } else if (n > 128) {
                    cftleaf(n, 1, a, offa, nw, w);
                } else {
                    cftfx41(n, a, offa, nw, w);
                }
                bitrv2conj(n, ip, a, offa);
            } else if (n == 32) {
                cftf161(a, offa, w, nw - 8);
                bitrv216neg(a, offa);
            } else {
                cftf081(a, offa, w, 0);
                bitrv208neg(a, offa);
            }
        } else if (n == 8) {
            cftb040(a, offa);
        } else if (n == 4) {
            cftxb020(a, offa);
        }
    }

    public static void bitrv2(int n, int[] ip, double[] a, int offa)
    {
        int j1, k1, l, m, nh, nm;
        double xr, xi, yr, yi;
        int idx0, idx1, idx2;

        m = 1;
        for (l = n >> 2; l > 8; l >>= 2) {
            m <<= 1;
        }
        nh = n >> 1;
        nm = 4 * m;
        if (l == 8) {
            for (int k = 0; k < m; k++) {
                idx0 = 4 * k;
                for (int j = 0; j < k; j++) {
                    j1 = 4 * j + 2 * ip[m + k];
                    k1 = idx0 + 2 * ip[m + j];
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nh;
                    k1 += 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += 2;
                    k1 += nh;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nh;
                    k1 -= 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                }
                k1 = idx0 + 2 * ip[m + k];
                j1 = k1 + 2;
                k1 += nh;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a[idx1];
                xi = a[idx1 + 1];
                yr = a[idx2];
                yi = a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                j1 += nm;
                k1 += 2 * nm;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a[idx1];
                xi = a[idx1 + 1];
                yr = a[idx2];
                yi = a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                j1 += nm;
                k1 -= nm;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a[idx1];
                xi = a[idx1 + 1];
                yr = a[idx2];
                yi = a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                j1 -= 2;
                k1 -= nh;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a[idx1];
                xi = a[idx1 + 1];
                yr = a[idx2];
                yi = a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                j1 += nh + 2;
                k1 += nh + 2;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a[idx1];
                xi = a[idx1 + 1];
                yr = a[idx2];
                yi = a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                j1 -= nh - nm;
                k1 += 2 * nm - 2;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a[idx1];
                xi = a[idx1 + 1];
                yr = a[idx2];
                yi = a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
            }
        } else {
            for (int k = 0; k < m; k++) {
                idx0 = 4 * k;
                for (int j = 0; j < k; j++) {
                    j1 = 4 * j + ip[m + k];
                    k1 = idx0 + ip[m + j];
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nh;
                    k1 += 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += 2;
                    k1 += nh;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nh;
                    k1 -= 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                }
                k1 = idx0 + ip[m + k];
                j1 = k1 + 2;
                k1 += nh;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a[idx1];
                xi = a[idx1 + 1];
                yr = a[idx2];
                yi = a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                j1 += nm;
                k1 += nm;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a[idx1];
                xi = a[idx1 + 1];
                yr = a[idx2];
                yi = a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
            }
        }
    }

    public static void bitrv2l(long n, LongLargeArray ip, DoubleLargeArray a, long offa)
    {
        long j1, k1, l, m, nh, nm;
        double xr, xi, yr, yi;
        long idx0, idx1, idx2;

        m = 1;
        for (l = n >> 2l; l > 8; l >>= 2l) {
            m <<= 1l;
        }
        nh = n >> 1l;
        nm = 4 * m;
        if (l == 8) {
            for (long k = 0; k < m; k++) {
                idx0 = 4 * k;
                for (long j = 0; j < k; j++) {
                    j1 = 4 * j + 2 * ip.getLong(m + k);
                    k1 = idx0 + 2 * ip.getLong(m + j);
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 += nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 += nh;
                    k1 += 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 -= nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 += 2;
                    k1 += nh;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 += nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 -= nh;
                    k1 -= 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 -= nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                }
                k1 = idx0 + 2 * ip.getLong(m + k);
                j1 = k1 + 2;
                k1 += nh;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a.getDouble(idx1);
                xi = a.getDouble(idx1 + 1);
                yr = a.getDouble(idx2);
                yi = a.getDouble(idx2 + 1);
                a.setDouble(idx1, yr);
                a.setDouble(idx1 + 1, yi);
                a.setDouble(idx2, xr);
                a.setDouble(idx2 + 1, xi);
                j1 += nm;
                k1 += 2 * nm;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a.getDouble(idx1);
                xi = a.getDouble(idx1 + 1);
                yr = a.getDouble(idx2);
                yi = a.getDouble(idx2 + 1);
                a.setDouble(idx1, yr);
                a.setDouble(idx1 + 1, yi);
                a.setDouble(idx2, xr);
                a.setDouble(idx2 + 1, xi);
                j1 += nm;
                k1 -= nm;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a.getDouble(idx1);
                xi = a.getDouble(idx1 + 1);
                yr = a.getDouble(idx2);
                yi = a.getDouble(idx2 + 1);
                a.setDouble(idx1, yr);
                a.setDouble(idx1 + 1, yi);
                a.setDouble(idx2, xr);
                a.setDouble(idx2 + 1, xi);
                j1 -= 2;
                k1 -= nh;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a.getDouble(idx1);
                xi = a.getDouble(idx1 + 1);
                yr = a.getDouble(idx2);
                yi = a.getDouble(idx2 + 1);
                a.setDouble(idx1, yr);
                a.setDouble(idx1 + 1, yi);
                a.setDouble(idx2, xr);
                a.setDouble(idx2 + 1, xi);
                j1 += nh + 2;
                k1 += nh + 2;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a.getDouble(idx1);
                xi = a.getDouble(idx1 + 1);
                yr = a.getDouble(idx2);
                yi = a.getDouble(idx2 + 1);
                a.setDouble(idx1, yr);
                a.setDouble(idx1 + 1, yi);
                a.setDouble(idx2, xr);
                a.setDouble(idx2 + 1, xi);
                j1 -= nh - nm;
                k1 += 2 * nm - 2;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a.getDouble(idx1);
                xi = a.getDouble(idx1 + 1);
                yr = a.getDouble(idx2);
                yi = a.getDouble(idx2 + 1);
                a.setDouble(idx1, yr);
                a.setDouble(idx1 + 1, yi);
                a.setDouble(idx2, xr);
                a.setDouble(idx2 + 1, xi);
            }
        } else {
            for (long k = 0; k < m; k++) {
                idx0 = 4 * k;
                for (long j = 0; j < k; j++) {
                    j1 = 4 * j + ip.getLong(m + k);
                    k1 = idx0 + ip.getLong(m + j);
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 += nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 += nh;
                    k1 += 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 -= nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 += 2;
                    k1 += nh;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 += nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 -= nh;
                    k1 -= 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 -= nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                }
                k1 = idx0 + ip.getLong(m + k);
                j1 = k1 + 2;
                k1 += nh;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a.getDouble(idx1);
                xi = a.getDouble(idx1 + 1);
                yr = a.getDouble(idx2);
                yi = a.getDouble(idx2 + 1);
                a.setDouble(idx1, yr);
                a.setDouble(idx1 + 1, yi);
                a.setDouble(idx2, xr);
                a.setDouble(idx2 + 1, xi);
                j1 += nm;
                k1 += nm;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a.getDouble(idx1);
                xi = a.getDouble(idx1 + 1);
                yr = a.getDouble(idx2);
                yi = a.getDouble(idx2 + 1);
                a.setDouble(idx1, yr);
                a.setDouble(idx1 + 1, yi);
                a.setDouble(idx2, xr);
                a.setDouble(idx2 + 1, xi);
            }
        }
    }

    public static void bitrv2conj(int n, int[] ip, double[] a, int offa)
    {
        int j1, k1, l, m, nh, nm;
        double xr, xi, yr, yi;
        int idx0, idx1, idx2;

        m = 1;
        for (l = n >> 2; l > 8; l >>= 2) {
            m <<= 1;
        }
        nh = n >> 1;
        nm = 4 * m;
        if (l == 8) {
            for (int k = 0; k < m; k++) {
                idx0 = 4 * k;
                for (int j = 0; j < k; j++) {
                    j1 = 4 * j + 2 * ip[m + k];
                    k1 = idx0 + 2 * ip[m + j];
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nh;
                    k1 += 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += 2;
                    k1 += nh;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nh;
                    k1 -= 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                }
                k1 = idx0 + 2 * ip[m + k];
                j1 = k1 + 2;
                k1 += nh;
                idx1 = offa + j1;
                idx2 = offa + k1;
                a[idx1 - 1] = -a[idx1 - 1];
                xr = a[idx1];
                xi = -a[idx1 + 1];
                yr = a[idx2];
                yi = -a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                a[idx2 + 3] = -a[idx2 + 3];
                j1 += nm;
                k1 += 2 * nm;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a[idx1];
                xi = -a[idx1 + 1];
                yr = a[idx2];
                yi = -a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                j1 += nm;
                k1 -= nm;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a[idx1];
                xi = -a[idx1 + 1];
                yr = a[idx2];
                yi = -a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                j1 -= 2;
                k1 -= nh;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a[idx1];
                xi = -a[idx1 + 1];
                yr = a[idx2];
                yi = -a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                j1 += nh + 2;
                k1 += nh + 2;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a[idx1];
                xi = -a[idx1 + 1];
                yr = a[idx2];
                yi = -a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                j1 -= nh - nm;
                k1 += 2 * nm - 2;
                idx1 = offa + j1;
                idx2 = offa + k1;
                a[idx1 - 1] = -a[idx1 - 1];
                xr = a[idx1];
                xi = -a[idx1 + 1];
                yr = a[idx2];
                yi = -a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                a[idx2 + 3] = -a[idx2 + 3];
            }
        } else {
            for (int k = 0; k < m; k++) {
                idx0 = 4 * k;
                for (int j = 0; j < k; j++) {
                    j1 = 4 * j + ip[m + k];
                    k1 = idx0 + ip[m + j];
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nh;
                    k1 += 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += 2;
                    k1 += nh;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nh;
                    k1 -= 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                }
                k1 = idx0 + ip[m + k];
                j1 = k1 + 2;
                k1 += nh;
                idx1 = offa + j1;
                idx2 = offa + k1;
                a[idx1 - 1] = -a[idx1 - 1];
                xr = a[idx1];
                xi = -a[idx1 + 1];
                yr = a[idx2];
                yi = -a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                a[idx2 + 3] = -a[idx2 + 3];
                j1 += nm;
                k1 += nm;
                idx1 = offa + j1;
                idx2 = offa + k1;
                a[idx1 - 1] = -a[idx1 - 1];
                xr = a[idx1];
                xi = -a[idx1 + 1];
                yr = a[idx2];
                yi = -a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                a[idx2 + 3] = -a[idx2 + 3];
            }
        }
    }

    public static void bitrv2conj(long n, LongLargeArray ip, DoubleLargeArray a, long offa)
    {
        long j1, k1, l, m, nh, nm;
        double xr, xi, yr, yi;
        long idx0, idx1, idx2;

        m = 1;
        for (l = n >> 2l; l > 8; l >>= 2l) {
            m <<= 1;
        }
        nh = n >> 1l;
        nm = 4 * m;
        if (l == 8) {
            for (long k = 0; k < m; k++) {
                idx0 = 4 * k;
                for (long j = 0; j < k; j++) {
                    j1 = 4 * j + 2 * ip.getLong(m + k);
                    k1 = idx0 + 2 * ip.getLong(m + j);
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = -a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = -a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = -a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = -a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 += nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = -a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = -a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = -a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = -a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 += nh;
                    k1 += 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = -a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = -a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = -a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = -a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 -= nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = -a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = -a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = -a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = -a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 += 2;
                    k1 += nh;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = -a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = -a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = -a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = -a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 += nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = -a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = -a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = -a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = -a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 -= nh;
                    k1 -= 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = -a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = -a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = -a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = -a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 -= nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = -a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = -a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = -a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = -a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                }
                k1 = idx0 + 2 * ip.getLong(m + k);
                j1 = k1 + 2;
                k1 += nh;
                idx1 = offa + j1;
                idx2 = offa + k1;
                a.setDouble(idx1 - 1, -a.getDouble(idx1 - 1));
                xr = a.getDouble(idx1);
                xi = -a.getDouble(idx1 + 1);
                yr = a.getDouble(idx2);
                yi = -a.getDouble(idx2 + 1);
                a.setDouble(idx1, yr);
                a.setDouble(idx1 + 1, yi);
                a.setDouble(idx2, xr);
                a.setDouble(idx2 + 1, xi);
                a.setDouble(idx2 + 3, -a.getDouble(idx2 + 3));
                j1 += nm;
                k1 += 2 * nm;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a.getDouble(idx1);
                xi = -a.getDouble(idx1 + 1);
                yr = a.getDouble(idx2);
                yi = -a.getDouble(idx2 + 1);
                a.setDouble(idx1, yr);
                a.setDouble(idx1 + 1, yi);
                a.setDouble(idx2, xr);
                a.setDouble(idx2 + 1, xi);
                j1 += nm;
                k1 -= nm;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a.getDouble(idx1);
                xi = -a.getDouble(idx1 + 1);
                yr = a.getDouble(idx2);
                yi = -a.getDouble(idx2 + 1);
                a.setDouble(idx1, yr);
                a.setDouble(idx1 + 1, yi);
                a.setDouble(idx2, xr);
                a.setDouble(idx2 + 1, xi);
                j1 -= 2;
                k1 -= nh;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a.getDouble(idx1);
                xi = -a.getDouble(idx1 + 1);
                yr = a.getDouble(idx2);
                yi = -a.getDouble(idx2 + 1);
                a.setDouble(idx1, yr);
                a.setDouble(idx1 + 1, yi);
                a.setDouble(idx2, xr);
                a.setDouble(idx2 + 1, xi);
                j1 += nh + 2;
                k1 += nh + 2;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a.getDouble(idx1);
                xi = -a.getDouble(idx1 + 1);
                yr = a.getDouble(idx2);
                yi = -a.getDouble(idx2 + 1);
                a.setDouble(idx1, yr);
                a.setDouble(idx1 + 1, yi);
                a.setDouble(idx2, xr);
                a.setDouble(idx2 + 1, xi);
                j1 -= nh - nm;
                k1 += 2 * nm - 2;
                idx1 = offa + j1;
                idx2 = offa + k1;
                a.setDouble(idx1 - 1, -a.getDouble(idx1 - 1));
                xr = a.getDouble(idx1);
                xi = -a.getDouble(idx1 + 1);
                yr = a.getDouble(idx2);
                yi = -a.getDouble(idx2 + 1);
                a.setDouble(idx1, yr);
                a.setDouble(idx1 + 1, yi);
                a.setDouble(idx2, xr);
                a.setDouble(idx2 + 1, xi);
                a.setDouble(idx2 + 3, -a.getDouble(idx2 + 3));
            }
        } else {
            for (int k = 0; k < m; k++) {
                idx0 = 4 * k;
                for (int j = 0; j < k; j++) {
                    j1 = 4 * j + ip.getLong(m + k);
                    k1 = idx0 + ip.getLong(m + j);
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = -a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = -a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 += nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = -a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = -a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 += nh;
                    k1 += 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = -a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = -a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 -= nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = -a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = -a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 += 2;
                    k1 += nh;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = -a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = -a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 += nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = -a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = -a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 -= nh;
                    k1 -= 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = -a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = -a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                    j1 -= nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getDouble(idx1);
                    xi = -a.getDouble(idx1 + 1);
                    yr = a.getDouble(idx2);
                    yi = -a.getDouble(idx2 + 1);
                    a.setDouble(idx1, yr);
                    a.setDouble(idx1 + 1, yi);
                    a.setDouble(idx2, xr);
                    a.setDouble(idx2 + 1, xi);
                }
                k1 = idx0 + ip.getLong(m + k);
                j1 = k1 + 2;
                k1 += nh;
                idx1 = offa + j1;
                idx2 = offa + k1;
                a.setDouble(idx1 - 1, -a.getDouble(idx1 - 1));
                xr = a.getDouble(idx1);
                xi = -a.getDouble(idx1 + 1);
                yr = a.getDouble(idx2);
                yi = -a.getDouble(idx2 + 1);
                a.setDouble(idx1, yr);
                a.setDouble(idx1 + 1, yi);
                a.setDouble(idx2, xr);
                a.setDouble(idx2 + 1, xi);
                a.setDouble(idx2 + 3, -a.getDouble(idx2 + 3));
                j1 += nm;
                k1 += nm;
                idx1 = offa + j1;
                idx2 = offa + k1;
                a.setDouble(idx1 - 1, -a.getDouble(idx1 - 1));
                xr = a.getDouble(idx1);
                xi = -a.getDouble(idx1 + 1);
                yr = a.getDouble(idx2);
                yi = -a.getDouble(idx2 + 1);
                a.setDouble(idx1, yr);
                a.setDouble(idx1 + 1, yi);
                a.setDouble(idx2, xr);
                a.setDouble(idx2 + 1, xi);
                a.setDouble(idx2 + 3, -a.getDouble(idx2 + 3));
            }
        }
    }

    public static void bitrv216(double[] a, int offa)
    {
        double x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, x5r, x5i, x7r, x7i, x8r, x8i, x10r, x10i, x11r, x11i, x12r, x12i, x13r, x13i, x14r, x14i;

        x1r = a[offa + 2];
        x1i = a[offa + 3];
        x2r = a[offa + 4];
        x2i = a[offa + 5];
        x3r = a[offa + 6];
        x3i = a[offa + 7];
        x4r = a[offa + 8];
        x4i = a[offa + 9];
        x5r = a[offa + 10];
        x5i = a[offa + 11];
        x7r = a[offa + 14];
        x7i = a[offa + 15];
        x8r = a[offa + 16];
        x8i = a[offa + 17];
        x10r = a[offa + 20];
        x10i = a[offa + 21];
        x11r = a[offa + 22];
        x11i = a[offa + 23];
        x12r = a[offa + 24];
        x12i = a[offa + 25];
        x13r = a[offa + 26];
        x13i = a[offa + 27];
        x14r = a[offa + 28];
        x14i = a[offa + 29];
        a[offa + 2] = x8r;
        a[offa + 3] = x8i;
        a[offa + 4] = x4r;
        a[offa + 5] = x4i;
        a[offa + 6] = x12r;
        a[offa + 7] = x12i;
        a[offa + 8] = x2r;
        a[offa + 9] = x2i;
        a[offa + 10] = x10r;
        a[offa + 11] = x10i;
        a[offa + 14] = x14r;
        a[offa + 15] = x14i;
        a[offa + 16] = x1r;
        a[offa + 17] = x1i;
        a[offa + 20] = x5r;
        a[offa + 21] = x5i;
        a[offa + 22] = x13r;
        a[offa + 23] = x13i;
        a[offa + 24] = x3r;
        a[offa + 25] = x3i;
        a[offa + 26] = x11r;
        a[offa + 27] = x11i;
        a[offa + 28] = x7r;
        a[offa + 29] = x7i;
    }

    public static void bitrv216(DoubleLargeArray a, long offa)
    {
        double x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, x5r, x5i, x7r, x7i, x8r, x8i, x10r, x10i, x11r, x11i, x12r, x12i, x13r, x13i, x14r, x14i;

        x1r = a.getDouble(offa + 2);
        x1i = a.getDouble(offa + 3);
        x2r = a.getDouble(offa + 4);
        x2i = a.getDouble(offa + 5);
        x3r = a.getDouble(offa + 6);
        x3i = a.getDouble(offa + 7);
        x4r = a.getDouble(offa + 8);
        x4i = a.getDouble(offa + 9);
        x5r = a.getDouble(offa + 10);
        x5i = a.getDouble(offa + 11);
        x7r = a.getDouble(offa + 14);
        x7i = a.getDouble(offa + 15);
        x8r = a.getDouble(offa + 16);
        x8i = a.getDouble(offa + 17);
        x10r = a.getDouble(offa + 20);
        x10i = a.getDouble(offa + 21);
        x11r = a.getDouble(offa + 22);
        x11i = a.getDouble(offa + 23);
        x12r = a.getDouble(offa + 24);
        x12i = a.getDouble(offa + 25);
        x13r = a.getDouble(offa + 26);
        x13i = a.getDouble(offa + 27);
        x14r = a.getDouble(offa + 28);
        x14i = a.getDouble(offa + 29);
        a.setDouble(offa + 2, x8r);
        a.setDouble(offa + 3, x8i);
        a.setDouble(offa + 4, x4r);
        a.setDouble(offa + 5, x4i);
        a.setDouble(offa + 6, x12r);
        a.setDouble(offa + 7, x12i);
        a.setDouble(offa + 8, x2r);
        a.setDouble(offa + 9, x2i);
        a.setDouble(offa + 10, x10r);
        a.setDouble(offa + 11, x10i);
        a.setDouble(offa + 14, x14r);
        a.setDouble(offa + 15, x14i);
        a.setDouble(offa + 16, x1r);
        a.setDouble(offa + 17, x1i);
        a.setDouble(offa + 20, x5r);
        a.setDouble(offa + 21, x5i);
        a.setDouble(offa + 22, x13r);
        a.setDouble(offa + 23, x13i);
        a.setDouble(offa + 24, x3r);
        a.setDouble(offa + 25, x3i);
        a.setDouble(offa + 26, x11r);
        a.setDouble(offa + 27, x11i);
        a.setDouble(offa + 28, x7r);
        a.setDouble(offa + 29, x7i);
    }

    public static void bitrv216neg(double[] a, int offa)
    {
        double x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i, x8r, x8i, x9r, x9i, x10r, x10i, x11r, x11i, x12r, x12i, x13r, x13i, x14r, x14i, x15r, x15i;

        x1r = a[offa + 2];
        x1i = a[offa + 3];
        x2r = a[offa + 4];
        x2i = a[offa + 5];
        x3r = a[offa + 6];
        x3i = a[offa + 7];
        x4r = a[offa + 8];
        x4i = a[offa + 9];
        x5r = a[offa + 10];
        x5i = a[offa + 11];
        x6r = a[offa + 12];
        x6i = a[offa + 13];
        x7r = a[offa + 14];
        x7i = a[offa + 15];
        x8r = a[offa + 16];
        x8i = a[offa + 17];
        x9r = a[offa + 18];
        x9i = a[offa + 19];
        x10r = a[offa + 20];
        x10i = a[offa + 21];
        x11r = a[offa + 22];
        x11i = a[offa + 23];
        x12r = a[offa + 24];
        x12i = a[offa + 25];
        x13r = a[offa + 26];
        x13i = a[offa + 27];
        x14r = a[offa + 28];
        x14i = a[offa + 29];
        x15r = a[offa + 30];
        x15i = a[offa + 31];
        a[offa + 2] = x15r;
        a[offa + 3] = x15i;
        a[offa + 4] = x7r;
        a[offa + 5] = x7i;
        a[offa + 6] = x11r;
        a[offa + 7] = x11i;
        a[offa + 8] = x3r;
        a[offa + 9] = x3i;
        a[offa + 10] = x13r;
        a[offa + 11] = x13i;
        a[offa + 12] = x5r;
        a[offa + 13] = x5i;
        a[offa + 14] = x9r;
        a[offa + 15] = x9i;
        a[offa + 16] = x1r;
        a[offa + 17] = x1i;
        a[offa + 18] = x14r;
        a[offa + 19] = x14i;
        a[offa + 20] = x6r;
        a[offa + 21] = x6i;
        a[offa + 22] = x10r;
        a[offa + 23] = x10i;
        a[offa + 24] = x2r;
        a[offa + 25] = x2i;
        a[offa + 26] = x12r;
        a[offa + 27] = x12i;
        a[offa + 28] = x4r;
        a[offa + 29] = x4i;
        a[offa + 30] = x8r;
        a[offa + 31] = x8i;
    }

    public static void bitrv216neg(DoubleLargeArray a, long offa)
    {
        double x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i, x8r, x8i, x9r, x9i, x10r, x10i, x11r, x11i, x12r, x12i, x13r, x13i, x14r, x14i, x15r, x15i;

        x1r = a.getDouble(offa + 2);
        x1i = a.getDouble(offa + 3);
        x2r = a.getDouble(offa + 4);
        x2i = a.getDouble(offa + 5);
        x3r = a.getDouble(offa + 6);
        x3i = a.getDouble(offa + 7);
        x4r = a.getDouble(offa + 8);
        x4i = a.getDouble(offa + 9);
        x5r = a.getDouble(offa + 10);
        x5i = a.getDouble(offa + 11);
        x6r = a.getDouble(offa + 12);
        x6i = a.getDouble(offa + 13);
        x7r = a.getDouble(offa + 14);
        x7i = a.getDouble(offa + 15);
        x8r = a.getDouble(offa + 16);
        x8i = a.getDouble(offa + 17);
        x9r = a.getDouble(offa + 18);
        x9i = a.getDouble(offa + 19);
        x10r = a.getDouble(offa + 20);
        x10i = a.getDouble(offa + 21);
        x11r = a.getDouble(offa + 22);
        x11i = a.getDouble(offa + 23);
        x12r = a.getDouble(offa + 24);
        x12i = a.getDouble(offa + 25);
        x13r = a.getDouble(offa + 26);
        x13i = a.getDouble(offa + 27);
        x14r = a.getDouble(offa + 28);
        x14i = a.getDouble(offa + 29);
        x15r = a.getDouble(offa + 30);
        x15i = a.getDouble(offa + 31);
        a.setDouble(offa + 2, x15r);
        a.setDouble(offa + 3, x15i);
        a.setDouble(offa + 4, x7r);
        a.setDouble(offa + 5, x7i);
        a.setDouble(offa + 6, x11r);
        a.setDouble(offa + 7, x11i);
        a.setDouble(offa + 8, x3r);
        a.setDouble(offa + 9, x3i);
        a.setDouble(offa + 10, x13r);
        a.setDouble(offa + 11, x13i);
        a.setDouble(offa + 12, x5r);
        a.setDouble(offa + 13, x5i);
        a.setDouble(offa + 14, x9r);
        a.setDouble(offa + 15, x9i);
        a.setDouble(offa + 16, x1r);
        a.setDouble(offa + 17, x1i);
        a.setDouble(offa + 18, x14r);
        a.setDouble(offa + 19, x14i);
        a.setDouble(offa + 20, x6r);
        a.setDouble(offa + 21, x6i);
        a.setDouble(offa + 22, x10r);
        a.setDouble(offa + 23, x10i);
        a.setDouble(offa + 24, x2r);
        a.setDouble(offa + 25, x2i);
        a.setDouble(offa + 26, x12r);
        a.setDouble(offa + 27, x12i);
        a.setDouble(offa + 28, x4r);
        a.setDouble(offa + 29, x4i);
        a.setDouble(offa + 30, x8r);
        a.setDouble(offa + 31, x8i);
    }

    public static void bitrv208(double[] a, int offa)
    {
        double x1r, x1i, x3r, x3i, x4r, x4i, x6r, x6i;

        x1r = a[offa + 2];
        x1i = a[offa + 3];
        x3r = a[offa + 6];
        x3i = a[offa + 7];
        x4r = a[offa + 8];
        x4i = a[offa + 9];
        x6r = a[offa + 12];
        x6i = a[offa + 13];
        a[offa + 2] = x4r;
        a[offa + 3] = x4i;
        a[offa + 6] = x6r;
        a[offa + 7] = x6i;
        a[offa + 8] = x1r;
        a[offa + 9] = x1i;
        a[offa + 12] = x3r;
        a[offa + 13] = x3i;
    }

    public static void bitrv208(DoubleLargeArray a, long offa)
    {
        double x1r, x1i, x3r, x3i, x4r, x4i, x6r, x6i;

        x1r = a.getDouble(offa + 2);
        x1i = a.getDouble(offa + 3);
        x3r = a.getDouble(offa + 6);
        x3i = a.getDouble(offa + 7);
        x4r = a.getDouble(offa + 8);
        x4i = a.getDouble(offa + 9);
        x6r = a.getDouble(offa + 12);
        x6i = a.getDouble(offa + 13);
        a.setDouble(offa + 2, x4r);
        a.setDouble(offa + 3, x4i);
        a.setDouble(offa + 6, x6r);
        a.setDouble(offa + 7, x6i);
        a.setDouble(offa + 8, x1r);
        a.setDouble(offa + 9, x1i);
        a.setDouble(offa + 12, x3r);
        a.setDouble(offa + 13, x3i);
    }

    public static void bitrv208neg(double[] a, int offa)
    {
        double x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i;

        x1r = a[offa + 2];
        x1i = a[offa + 3];
        x2r = a[offa + 4];
        x2i = a[offa + 5];
        x3r = a[offa + 6];
        x3i = a[offa + 7];
        x4r = a[offa + 8];
        x4i = a[offa + 9];
        x5r = a[offa + 10];
        x5i = a[offa + 11];
        x6r = a[offa + 12];
        x6i = a[offa + 13];
        x7r = a[offa + 14];
        x7i = a[offa + 15];
        a[offa + 2] = x7r;
        a[offa + 3] = x7i;
        a[offa + 4] = x3r;
        a[offa + 5] = x3i;
        a[offa + 6] = x5r;
        a[offa + 7] = x5i;
        a[offa + 8] = x1r;
        a[offa + 9] = x1i;
        a[offa + 10] = x6r;
        a[offa + 11] = x6i;
        a[offa + 12] = x2r;
        a[offa + 13] = x2i;
        a[offa + 14] = x4r;
        a[offa + 15] = x4i;
    }

    public static void bitrv208neg(DoubleLargeArray a, long offa)
    {
        double x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i;

        x1r = a.getDouble(offa + 2);
        x1i = a.getDouble(offa + 3);
        x2r = a.getDouble(offa + 4);
        x2i = a.getDouble(offa + 5);
        x3r = a.getDouble(offa + 6);
        x3i = a.getDouble(offa + 7);
        x4r = a.getDouble(offa + 8);
        x4i = a.getDouble(offa + 9);
        x5r = a.getDouble(offa + 10);
        x5i = a.getDouble(offa + 11);
        x6r = a.getDouble(offa + 12);
        x6i = a.getDouble(offa + 13);
        x7r = a.getDouble(offa + 14);
        x7i = a.getDouble(offa + 15);
        a.setDouble(offa + 2, x7r);
        a.setDouble(offa + 3, x7i);
        a.setDouble(offa + 4, x3r);
        a.setDouble(offa + 5, x3i);
        a.setDouble(offa + 6, x5r);
        a.setDouble(offa + 7, x5i);
        a.setDouble(offa + 8, x1r);
        a.setDouble(offa + 9, x1i);
        a.setDouble(offa + 10, x6r);
        a.setDouble(offa + 11, x6i);
        a.setDouble(offa + 12, x2r);
        a.setDouble(offa + 13, x2i);
        a.setDouble(offa + 14, x4r);
        a.setDouble(offa + 15, x4i);
    }

    public static void cftf1st(int n, double[] a, int offa, double[] w, int startw)
    {
        int j0, j1, j2, j3, k, m, mh;
        double wn4r, csc1, csc3, wk1r, wk1i, wk3r, wk3i, wd1r, wd1i, wd3r, wd3i;
        double x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        int idx0, idx1, idx2, idx3, idx4, idx5;
        mh = n >> 3;
        m = 2 * mh;
        j1 = m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a[offa] + a[idx2];
        x0i = a[offa + 1] + a[idx2 + 1];
        x1r = a[offa] - a[idx2];
        x1i = a[offa + 1] - a[idx2 + 1];
        x2r = a[idx1] + a[idx3];
        x2i = a[idx1 + 1] + a[idx3 + 1];
        x3r = a[idx1] - a[idx3];
        x3i = a[idx1 + 1] - a[idx3 + 1];
        a[offa] = x0r + x2r;
        a[offa + 1] = x0i + x2i;
        a[idx1] = x0r - x2r;
        a[idx1 + 1] = x0i - x2i;
        a[idx2] = x1r - x3i;
        a[idx2 + 1] = x1i + x3r;
        a[idx3] = x1r + x3i;
        a[idx3 + 1] = x1i - x3r;
        wn4r = w[startw + 1];
        csc1 = w[startw + 2];
        csc3 = w[startw + 3];
        wd1r = 1;
        wd1i = 0;
        wd3r = 1;
        wd3i = 0;
        k = 0;
        for (int j = 2; j < mh - 2; j += 4) {
            k += 4;
            idx4 = startw + k;
            wk1r = csc1 * (wd1r + w[idx4]);
            wk1i = csc1 * (wd1i + w[idx4 + 1]);
            wk3r = csc3 * (wd3r + w[idx4 + 2]);
            wk3i = csc3 * (wd3i + w[idx4 + 3]);
            wd1r = w[idx4];
            wd1i = w[idx4 + 1];
            wd3r = w[idx4 + 2];
            wd3i = w[idx4 + 3];
            j1 = j + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            idx5 = offa + j;
            x0r = a[idx5] + a[idx2];
            x0i = a[idx5 + 1] + a[idx2 + 1];
            x1r = a[idx5] - a[idx2];
            x1i = a[idx5 + 1] - a[idx2 + 1];
            y0r = a[idx5 + 2] + a[idx2 + 2];
            y0i = a[idx5 + 3] + a[idx2 + 3];
            y1r = a[idx5 + 2] - a[idx2 + 2];
            y1i = a[idx5 + 3] - a[idx2 + 3];
            x2r = a[idx1] + a[idx3];
            x2i = a[idx1 + 1] + a[idx3 + 1];
            x3r = a[idx1] - a[idx3];
            x3i = a[idx1 + 1] - a[idx3 + 1];
            y2r = a[idx1 + 2] + a[idx3 + 2];
            y2i = a[idx1 + 3] + a[idx3 + 3];
            y3r = a[idx1 + 2] - a[idx3 + 2];
            y3i = a[idx1 + 3] - a[idx3 + 3];
            a[idx5] = x0r + x2r;
            a[idx5 + 1] = x0i + x2i;
            a[idx5 + 2] = y0r + y2r;
            a[idx5 + 3] = y0i + y2i;
            a[idx1] = x0r - x2r;
            a[idx1 + 1] = x0i - x2i;
            a[idx1 + 2] = y0r - y2r;
            a[idx1 + 3] = y0i - y2i;
            x0r = x1r - x3i;
            x0i = x1i + x3r;
            a[idx2] = wk1r * x0r - wk1i * x0i;
            a[idx2 + 1] = wk1r * x0i + wk1i * x0r;
            x0r = y1r - y3i;
            x0i = y1i + y3r;
            a[idx2 + 2] = wd1r * x0r - wd1i * x0i;
            a[idx2 + 3] = wd1r * x0i + wd1i * x0r;
            x0r = x1r + x3i;
            x0i = x1i - x3r;
            a[idx3] = wk3r * x0r + wk3i * x0i;
            a[idx3 + 1] = wk3r * x0i - wk3i * x0r;
            x0r = y1r + y3i;
            x0i = y1i - y3r;
            a[idx3 + 2] = wd3r * x0r + wd3i * x0i;
            a[idx3 + 3] = wd3r * x0i - wd3i * x0r;
            j0 = m - j;
            j1 = j0 + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx0 = offa + j0;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            x0r = a[idx0] + a[idx2];
            x0i = a[idx0 + 1] + a[idx2 + 1];
            x1r = a[idx0] - a[idx2];
            x1i = a[idx0 + 1] - a[idx2 + 1];
            y0r = a[idx0 - 2] + a[idx2 - 2];
            y0i = a[idx0 - 1] + a[idx2 - 1];
            y1r = a[idx0 - 2] - a[idx2 - 2];
            y1i = a[idx0 - 1] - a[idx2 - 1];
            x2r = a[idx1] + a[idx3];
            x2i = a[idx1 + 1] + a[idx3 + 1];
            x3r = a[idx1] - a[idx3];
            x3i = a[idx1 + 1] - a[idx3 + 1];
            y2r = a[idx1 - 2] + a[idx3 - 2];
            y2i = a[idx1 - 1] + a[idx3 - 1];
            y3r = a[idx1 - 2] - a[idx3 - 2];
            y3i = a[idx1 - 1] - a[idx3 - 1];
            a[idx0] = x0r + x2r;
            a[idx0 + 1] = x0i + x2i;
            a[idx0 - 2] = y0r + y2r;
            a[idx0 - 1] = y0i + y2i;
            a[idx1] = x0r - x2r;
            a[idx1 + 1] = x0i - x2i;
            a[idx1 - 2] = y0r - y2r;
            a[idx1 - 1] = y0i - y2i;
            x0r = x1r - x3i;
            x0i = x1i + x3r;
            a[idx2] = wk1i * x0r - wk1r * x0i;
            a[idx2 + 1] = wk1i * x0i + wk1r * x0r;
            x0r = y1r - y3i;
            x0i = y1i + y3r;
            a[idx2 - 2] = wd1i * x0r - wd1r * x0i;
            a[idx2 - 1] = wd1i * x0i + wd1r * x0r;
            x0r = x1r + x3i;
            x0i = x1i - x3r;
            a[idx3] = wk3i * x0r + wk3r * x0i;
            a[idx3 + 1] = wk3i * x0i - wk3r * x0r;
            x0r = y1r + y3i;
            x0i = y1i - y3r;
            a[offa + j3 - 2] = wd3i * x0r + wd3r * x0i;
            a[offa + j3 - 1] = wd3i * x0i - wd3r * x0r;
        }
        wk1r = csc1 * (wd1r + wn4r);
        wk1i = csc1 * (wd1i + wn4r);
        wk3r = csc3 * (wd3r - wn4r);
        wk3i = csc3 * (wd3i - wn4r);
        j0 = mh;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx0 = offa + j0;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a[idx0 - 2] + a[idx2 - 2];
        x0i = a[idx0 - 1] + a[idx2 - 1];
        x1r = a[idx0 - 2] - a[idx2 - 2];
        x1i = a[idx0 - 1] - a[idx2 - 1];
        x2r = a[idx1 - 2] + a[idx3 - 2];
        x2i = a[idx1 - 1] + a[idx3 - 1];
        x3r = a[idx1 - 2] - a[idx3 - 2];
        x3i = a[idx1 - 1] - a[idx3 - 1];
        a[idx0 - 2] = x0r + x2r;
        a[idx0 - 1] = x0i + x2i;
        a[idx1 - 2] = x0r - x2r;
        a[idx1 - 1] = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a[idx2 - 2] = wk1r * x0r - wk1i * x0i;
        a[idx2 - 1] = wk1r * x0i + wk1i * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a[idx3 - 2] = wk3r * x0r + wk3i * x0i;
        a[idx3 - 1] = wk3r * x0i - wk3i * x0r;
        x0r = a[idx0] + a[idx2];
        x0i = a[idx0 + 1] + a[idx2 + 1];
        x1r = a[idx0] - a[idx2];
        x1i = a[idx0 + 1] - a[idx2 + 1];
        x2r = a[idx1] + a[idx3];
        x2i = a[idx1 + 1] + a[idx3 + 1];
        x3r = a[idx1] - a[idx3];
        x3i = a[idx1 + 1] - a[idx3 + 1];
        a[idx0] = x0r + x2r;
        a[idx0 + 1] = x0i + x2i;
        a[idx1] = x0r - x2r;
        a[idx1 + 1] = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a[idx2] = wn4r * (x0r - x0i);
        a[idx2 + 1] = wn4r * (x0i + x0r);
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a[idx3] = -wn4r * (x0r + x0i);
        a[idx3 + 1] = -wn4r * (x0i - x0r);
        x0r = a[idx0 + 2] + a[idx2 + 2];
        x0i = a[idx0 + 3] + a[idx2 + 3];
        x1r = a[idx0 + 2] - a[idx2 + 2];
        x1i = a[idx0 + 3] - a[idx2 + 3];
        x2r = a[idx1 + 2] + a[idx3 + 2];
        x2i = a[idx1 + 3] + a[idx3 + 3];
        x3r = a[idx1 + 2] - a[idx3 + 2];
        x3i = a[idx1 + 3] - a[idx3 + 3];
        a[idx0 + 2] = x0r + x2r;
        a[idx0 + 3] = x0i + x2i;
        a[idx1 + 2] = x0r - x2r;
        a[idx1 + 3] = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a[idx2 + 2] = wk1i * x0r - wk1r * x0i;
        a[idx2 + 3] = wk1i * x0i + wk1r * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a[idx3 + 2] = wk3i * x0r + wk3r * x0i;
        a[idx3 + 3] = wk3i * x0i - wk3r * x0r;
    }

    public static void cftf1st(long n, DoubleLargeArray a, long offa, DoubleLargeArray w, long startw)
    {
        long j0, j1, j2, j3, k, m, mh;
        double wn4r, csc1, csc3, wk1r, wk1i, wk3r, wk3i, wd1r, wd1i, wd3r, wd3i;
        double x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        long idx0, idx1, idx2, idx3, idx4, idx5;
        mh = n >> 3l;
        m = 2 * mh;
        j1 = m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a.getDouble(offa) + a.getDouble(idx2);
        x0i = a.getDouble(offa + 1) + a.getDouble(idx2 + 1);
        x1r = a.getDouble(offa) - a.getDouble(idx2);
        x1i = a.getDouble(offa + 1) - a.getDouble(idx2 + 1);
        x2r = a.getDouble(idx1) + a.getDouble(idx3);
        x2i = a.getDouble(idx1 + 1) + a.getDouble(idx3 + 1);
        x3r = a.getDouble(idx1) - a.getDouble(idx3);
        x3i = a.getDouble(idx1 + 1) - a.getDouble(idx3 + 1);
        a.setDouble(offa, x0r + x2r);
        a.setDouble(offa + 1, x0i + x2i);
        a.setDouble(idx1, x0r - x2r);
        a.setDouble(idx1 + 1, x0i - x2i);
        a.setDouble(idx2, x1r - x3i);
        a.setDouble(idx2 + 1, x1i + x3r);
        a.setDouble(idx3, x1r + x3i);
        a.setDouble(idx3 + 1, x1i - x3r);
        wn4r = w.getDouble(startw + 1);
        csc1 = w.getDouble(startw + 2);
        csc3 = w.getDouble(startw + 3);
        wd1r = 1;
        wd1i = 0;
        wd3r = 1;
        wd3i = 0;
        k = 0;
        for (int j = 2; j < mh - 2; j += 4) {
            k += 4;
            idx4 = startw + k;
            wk1r = csc1 * (wd1r + w.getDouble(idx4));
            wk1i = csc1 * (wd1i + w.getDouble(idx4 + 1));
            wk3r = csc3 * (wd3r + w.getDouble(idx4 + 2));
            wk3i = csc3 * (wd3i + w.getDouble(idx4 + 3));
            wd1r = w.getDouble(idx4);
            wd1i = w.getDouble(idx4 + 1);
            wd3r = w.getDouble(idx4 + 2);
            wd3i = w.getDouble(idx4 + 3);
            j1 = j + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            idx5 = offa + j;
            x0r = a.getDouble(idx5) + a.getDouble(idx2);
            x0i = a.getDouble(idx5 + 1) + a.getDouble(idx2 + 1);
            x1r = a.getDouble(idx5) - a.getDouble(idx2);
            x1i = a.getDouble(idx5 + 1) - a.getDouble(idx2 + 1);
            y0r = a.getDouble(idx5 + 2) + a.getDouble(idx2 + 2);
            y0i = a.getDouble(idx5 + 3) + a.getDouble(idx2 + 3);
            y1r = a.getDouble(idx5 + 2) - a.getDouble(idx2 + 2);
            y1i = a.getDouble(idx5 + 3) - a.getDouble(idx2 + 3);
            x2r = a.getDouble(idx1) + a.getDouble(idx3);
            x2i = a.getDouble(idx1 + 1) + a.getDouble(idx3 + 1);
            x3r = a.getDouble(idx1) - a.getDouble(idx3);
            x3i = a.getDouble(idx1 + 1) - a.getDouble(idx3 + 1);
            y2r = a.getDouble(idx1 + 2) + a.getDouble(idx3 + 2);
            y2i = a.getDouble(idx1 + 3) + a.getDouble(idx3 + 3);
            y3r = a.getDouble(idx1 + 2) - a.getDouble(idx3 + 2);
            y3i = a.getDouble(idx1 + 3) - a.getDouble(idx3 + 3);
            a.setDouble(idx5, x0r + x2r);
            a.setDouble(idx5 + 1, x0i + x2i);
            a.setDouble(idx5 + 2, y0r + y2r);
            a.setDouble(idx5 + 3, y0i + y2i);
            a.setDouble(idx1, x0r - x2r);
            a.setDouble(idx1 + 1, x0i - x2i);
            a.setDouble(idx1 + 2, y0r - y2r);
            a.setDouble(idx1 + 3, y0i - y2i);
            x0r = x1r - x3i;
            x0i = x1i + x3r;
            a.setDouble(idx2, wk1r * x0r - wk1i * x0i);
            a.setDouble(idx2 + 1, wk1r * x0i + wk1i * x0r);
            x0r = y1r - y3i;
            x0i = y1i + y3r;
            a.setDouble(idx2 + 2, wd1r * x0r - wd1i * x0i);
            a.setDouble(idx2 + 3, wd1r * x0i + wd1i * x0r);
            x0r = x1r + x3i;
            x0i = x1i - x3r;
            a.setDouble(idx3, wk3r * x0r + wk3i * x0i);
            a.setDouble(idx3 + 1, wk3r * x0i - wk3i * x0r);
            x0r = y1r + y3i;
            x0i = y1i - y3r;
            a.setDouble(idx3 + 2, wd3r * x0r + wd3i * x0i);
            a.setDouble(idx3 + 3, wd3r * x0i - wd3i * x0r);
            j0 = m - j;
            j1 = j0 + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx0 = offa + j0;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            x0r = a.getDouble(idx0) + a.getDouble(idx2);
            x0i = a.getDouble(idx0 + 1) + a.getDouble(idx2 + 1);
            x1r = a.getDouble(idx0) - a.getDouble(idx2);
            x1i = a.getDouble(idx0 + 1) - a.getDouble(idx2 + 1);
            y0r = a.getDouble(idx0 - 2) + a.getDouble(idx2 - 2);
            y0i = a.getDouble(idx0 - 1) + a.getDouble(idx2 - 1);
            y1r = a.getDouble(idx0 - 2) - a.getDouble(idx2 - 2);
            y1i = a.getDouble(idx0 - 1) - a.getDouble(idx2 - 1);
            x2r = a.getDouble(idx1) + a.getDouble(idx3);
            x2i = a.getDouble(idx1 + 1) + a.getDouble(idx3 + 1);
            x3r = a.getDouble(idx1) - a.getDouble(idx3);
            x3i = a.getDouble(idx1 + 1) - a.getDouble(idx3 + 1);
            y2r = a.getDouble(idx1 - 2) + a.getDouble(idx3 - 2);
            y2i = a.getDouble(idx1 - 1) + a.getDouble(idx3 - 1);
            y3r = a.getDouble(idx1 - 2) - a.getDouble(idx3 - 2);
            y3i = a.getDouble(idx1 - 1) - a.getDouble(idx3 - 1);
            a.setDouble(idx0, x0r + x2r);
            a.setDouble(idx0 + 1, x0i + x2i);
            a.setDouble(idx0 - 2, y0r + y2r);
            a.setDouble(idx0 - 1, y0i + y2i);
            a.setDouble(idx1, x0r - x2r);
            a.setDouble(idx1 + 1, x0i - x2i);
            a.setDouble(idx1 - 2, y0r - y2r);
            a.setDouble(idx1 - 1, y0i - y2i);
            x0r = x1r - x3i;
            x0i = x1i + x3r;
            a.setDouble(idx2, wk1i * x0r - wk1r * x0i);
            a.setDouble(idx2 + 1, wk1i * x0i + wk1r * x0r);
            x0r = y1r - y3i;
            x0i = y1i + y3r;
            a.setDouble(idx2 - 2, wd1i * x0r - wd1r * x0i);
            a.setDouble(idx2 - 1, wd1i * x0i + wd1r * x0r);
            x0r = x1r + x3i;
            x0i = x1i - x3r;
            a.setDouble(idx3, wk3i * x0r + wk3r * x0i);
            a.setDouble(idx3 + 1, wk3i * x0i - wk3r * x0r);
            x0r = y1r + y3i;
            x0i = y1i - y3r;
            a.setDouble(offa + j3 - 2, wd3i * x0r + wd3r * x0i);
            a.setDouble(offa + j3 - 1, wd3i * x0i - wd3r * x0r);
        }
        wk1r = csc1 * (wd1r + wn4r);
        wk1i = csc1 * (wd1i + wn4r);
        wk3r = csc3 * (wd3r - wn4r);
        wk3i = csc3 * (wd3i - wn4r);
        j0 = mh;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx0 = offa + j0;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a.getDouble(idx0 - 2) + a.getDouble(idx2 - 2);
        x0i = a.getDouble(idx0 - 1) + a.getDouble(idx2 - 1);
        x1r = a.getDouble(idx0 - 2) - a.getDouble(idx2 - 2);
        x1i = a.getDouble(idx0 - 1) - a.getDouble(idx2 - 1);
        x2r = a.getDouble(idx1 - 2) + a.getDouble(idx3 - 2);
        x2i = a.getDouble(idx1 - 1) + a.getDouble(idx3 - 1);
        x3r = a.getDouble(idx1 - 2) - a.getDouble(idx3 - 2);
        x3i = a.getDouble(idx1 - 1) - a.getDouble(idx3 - 1);
        a.setDouble(idx0 - 2, x0r + x2r);
        a.setDouble(idx0 - 1, x0i + x2i);
        a.setDouble(idx1 - 2, x0r - x2r);
        a.setDouble(idx1 - 1, x0i - x2i);
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a.setDouble(idx2 - 2, wk1r * x0r - wk1i * x0i);
        a.setDouble(idx2 - 1, wk1r * x0i + wk1i * x0r);
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a.setDouble(idx3 - 2, wk3r * x0r + wk3i * x0i);
        a.setDouble(idx3 - 1, wk3r * x0i - wk3i * x0r);
        x0r = a.getDouble(idx0) + a.getDouble(idx2);
        x0i = a.getDouble(idx0 + 1) + a.getDouble(idx2 + 1);
        x1r = a.getDouble(idx0) - a.getDouble(idx2);
        x1i = a.getDouble(idx0 + 1) - a.getDouble(idx2 + 1);
        x2r = a.getDouble(idx1) + a.getDouble(idx3);
        x2i = a.getDouble(idx1 + 1) + a.getDouble(idx3 + 1);
        x3r = a.getDouble(idx1) - a.getDouble(idx3);
        x3i = a.getDouble(idx1 + 1) - a.getDouble(idx3 + 1);
        a.setDouble(idx0, x0r + x2r);
        a.setDouble(idx0 + 1, x0i + x2i);
        a.setDouble(idx1, x0r - x2r);
        a.setDouble(idx1 + 1, x0i - x2i);
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a.setDouble(idx2, wn4r * (x0r - x0i));
        a.setDouble(idx2 + 1, wn4r * (x0i + x0r));
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a.setDouble(idx3, -wn4r * (x0r + x0i));
        a.setDouble(idx3 + 1, -wn4r * (x0i - x0r));
        x0r = a.getDouble(idx0 + 2) + a.getDouble(idx2 + 2);
        x0i = a.getDouble(idx0 + 3) + a.getDouble(idx2 + 3);
        x1r = a.getDouble(idx0 + 2) - a.getDouble(idx2 + 2);
        x1i = a.getDouble(idx0 + 3) - a.getDouble(idx2 + 3);
        x2r = a.getDouble(idx1 + 2) + a.getDouble(idx3 + 2);
        x2i = a.getDouble(idx1 + 3) + a.getDouble(idx3 + 3);
        x3r = a.getDouble(idx1 + 2) - a.getDouble(idx3 + 2);
        x3i = a.getDouble(idx1 + 3) - a.getDouble(idx3 + 3);
        a.setDouble(idx0 + 2, x0r + x2r);
        a.setDouble(idx0 + 3, x0i + x2i);
        a.setDouble(idx1 + 2, x0r - x2r);
        a.setDouble(idx1 + 3, x0i - x2i);
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a.setDouble(idx2 + 2, wk1i * x0r - wk1r * x0i);
        a.setDouble(idx2 + 3, wk1i * x0i + wk1r * x0r);
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a.setDouble(idx3 + 2, wk3i * x0r + wk3r * x0i);
        a.setDouble(idx3 + 3, wk3i * x0i - wk3r * x0r);
    }

    public static void cftb1st(int n, double[] a, int offa, double[] w, int startw)
    {
        int j0, j1, j2, j3, k, m, mh;
        double wn4r, csc1, csc3, wk1r, wk1i, wk3r, wk3i, wd1r, wd1i, wd3r, wd3i;
        double x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        int idx0, idx1, idx2, idx3, idx4, idx5;
        mh = n >> 3;
        m = 2 * mh;
        j1 = m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;

        x0r = a[offa] + a[idx2];
        x0i = -a[offa + 1] - a[idx2 + 1];
        x1r = a[offa] - a[idx2];
        x1i = -a[offa + 1] + a[idx2 + 1];
        x2r = a[idx1] + a[idx3];
        x2i = a[idx1 + 1] + a[idx3 + 1];
        x3r = a[idx1] - a[idx3];
        x3i = a[idx1 + 1] - a[idx3 + 1];
        a[offa] = x0r + x2r;
        a[offa + 1] = x0i - x2i;
        a[idx1] = x0r - x2r;
        a[idx1 + 1] = x0i + x2i;
        a[idx2] = x1r + x3i;
        a[idx2 + 1] = x1i + x3r;
        a[idx3] = x1r - x3i;
        a[idx3 + 1] = x1i - x3r;
        wn4r = w[startw + 1];
        csc1 = w[startw + 2];
        csc3 = w[startw + 3];
        wd1r = 1;
        wd1i = 0;
        wd3r = 1;
        wd3i = 0;
        k = 0;
        for (int j = 2; j < mh - 2; j += 4) {
            k += 4;
            idx4 = startw + k;
            wk1r = csc1 * (wd1r + w[idx4]);
            wk1i = csc1 * (wd1i + w[idx4 + 1]);
            wk3r = csc3 * (wd3r + w[idx4 + 2]);
            wk3i = csc3 * (wd3i + w[idx4 + 3]);
            wd1r = w[idx4];
            wd1i = w[idx4 + 1];
            wd3r = w[idx4 + 2];
            wd3i = w[idx4 + 3];
            j1 = j + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            idx5 = offa + j;
            x0r = a[idx5] + a[idx2];
            x0i = -a[idx5 + 1] - a[idx2 + 1];
            x1r = a[idx5] - a[offa + j2];
            x1i = -a[idx5 + 1] + a[idx2 + 1];
            y0r = a[idx5 + 2] + a[idx2 + 2];
            y0i = -a[idx5 + 3] - a[idx2 + 3];
            y1r = a[idx5 + 2] - a[idx2 + 2];
            y1i = -a[idx5 + 3] + a[idx2 + 3];
            x2r = a[idx1] + a[idx3];
            x2i = a[idx1 + 1] + a[idx3 + 1];
            x3r = a[idx1] - a[idx3];
            x3i = a[idx1 + 1] - a[idx3 + 1];
            y2r = a[idx1 + 2] + a[idx3 + 2];
            y2i = a[idx1 + 3] + a[idx3 + 3];
            y3r = a[idx1 + 2] - a[idx3 + 2];
            y3i = a[idx1 + 3] - a[idx3 + 3];
            a[idx5] = x0r + x2r;
            a[idx5 + 1] = x0i - x2i;
            a[idx5 + 2] = y0r + y2r;
            a[idx5 + 3] = y0i - y2i;
            a[idx1] = x0r - x2r;
            a[idx1 + 1] = x0i + x2i;
            a[idx1 + 2] = y0r - y2r;
            a[idx1 + 3] = y0i + y2i;
            x0r = x1r + x3i;
            x0i = x1i + x3r;
            a[idx2] = wk1r * x0r - wk1i * x0i;
            a[idx2 + 1] = wk1r * x0i + wk1i * x0r;
            x0r = y1r + y3i;
            x0i = y1i + y3r;
            a[idx2 + 2] = wd1r * x0r - wd1i * x0i;
            a[idx2 + 3] = wd1r * x0i + wd1i * x0r;
            x0r = x1r - x3i;
            x0i = x1i - x3r;
            a[idx3] = wk3r * x0r + wk3i * x0i;
            a[idx3 + 1] = wk3r * x0i - wk3i * x0r;
            x0r = y1r - y3i;
            x0i = y1i - y3r;
            a[idx3 + 2] = wd3r * x0r + wd3i * x0i;
            a[idx3 + 3] = wd3r * x0i - wd3i * x0r;
            j0 = m - j;
            j1 = j0 + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx0 = offa + j0;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            x0r = a[idx0] + a[idx2];
            x0i = -a[idx0 + 1] - a[idx2 + 1];
            x1r = a[idx0] - a[idx2];
            x1i = -a[idx0 + 1] + a[idx2 + 1];
            y0r = a[idx0 - 2] + a[idx2 - 2];
            y0i = -a[idx0 - 1] - a[idx2 - 1];
            y1r = a[idx0 - 2] - a[idx2 - 2];
            y1i = -a[idx0 - 1] + a[idx2 - 1];
            x2r = a[idx1] + a[idx3];
            x2i = a[idx1 + 1] + a[idx3 + 1];
            x3r = a[idx1] - a[idx3];
            x3i = a[idx1 + 1] - a[idx3 + 1];
            y2r = a[idx1 - 2] + a[idx3 - 2];
            y2i = a[idx1 - 1] + a[idx3 - 1];
            y3r = a[idx1 - 2] - a[idx3 - 2];
            y3i = a[idx1 - 1] - a[idx3 - 1];
            a[idx0] = x0r + x2r;
            a[idx0 + 1] = x0i - x2i;
            a[idx0 - 2] = y0r + y2r;
            a[idx0 - 1] = y0i - y2i;
            a[idx1] = x0r - x2r;
            a[idx1 + 1] = x0i + x2i;
            a[idx1 - 2] = y0r - y2r;
            a[idx1 - 1] = y0i + y2i;
            x0r = x1r + x3i;
            x0i = x1i + x3r;
            a[idx2] = wk1i * x0r - wk1r * x0i;
            a[idx2 + 1] = wk1i * x0i + wk1r * x0r;
            x0r = y1r + y3i;
            x0i = y1i + y3r;
            a[idx2 - 2] = wd1i * x0r - wd1r * x0i;
            a[idx2 - 1] = wd1i * x0i + wd1r * x0r;
            x0r = x1r - x3i;
            x0i = x1i - x3r;
            a[idx3] = wk3i * x0r + wk3r * x0i;
            a[idx3 + 1] = wk3i * x0i - wk3r * x0r;
            x0r = y1r - y3i;
            x0i = y1i - y3r;
            a[idx3 - 2] = wd3i * x0r + wd3r * x0i;
            a[idx3 - 1] = wd3i * x0i - wd3r * x0r;
        }
        wk1r = csc1 * (wd1r + wn4r);
        wk1i = csc1 * (wd1i + wn4r);
        wk3r = csc3 * (wd3r - wn4r);
        wk3i = csc3 * (wd3i - wn4r);
        j0 = mh;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx0 = offa + j0;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a[idx0 - 2] + a[idx2 - 2];
        x0i = -a[idx0 - 1] - a[idx2 - 1];
        x1r = a[idx0 - 2] - a[idx2 - 2];
        x1i = -a[idx0 - 1] + a[idx2 - 1];
        x2r = a[idx1 - 2] + a[idx3 - 2];
        x2i = a[idx1 - 1] + a[idx3 - 1];
        x3r = a[idx1 - 2] - a[idx3 - 2];
        x3i = a[idx1 - 1] - a[idx3 - 1];
        a[idx0 - 2] = x0r + x2r;
        a[idx0 - 1] = x0i - x2i;
        a[idx1 - 2] = x0r - x2r;
        a[idx1 - 1] = x0i + x2i;
        x0r = x1r + x3i;
        x0i = x1i + x3r;
        a[idx2 - 2] = wk1r * x0r - wk1i * x0i;
        a[idx2 - 1] = wk1r * x0i + wk1i * x0r;
        x0r = x1r - x3i;
        x0i = x1i - x3r;
        a[idx3 - 2] = wk3r * x0r + wk3i * x0i;
        a[idx3 - 1] = wk3r * x0i - wk3i * x0r;
        x0r = a[idx0] + a[idx2];
        x0i = -a[idx0 + 1] - a[idx2 + 1];
        x1r = a[idx0] - a[idx2];
        x1i = -a[idx0 + 1] + a[idx2 + 1];
        x2r = a[idx1] + a[idx3];
        x2i = a[idx1 + 1] + a[idx3 + 1];
        x3r = a[idx1] - a[idx3];
        x3i = a[idx1 + 1] - a[idx3 + 1];
        a[idx0] = x0r + x2r;
        a[idx0 + 1] = x0i - x2i;
        a[idx1] = x0r - x2r;
        a[idx1 + 1] = x0i + x2i;
        x0r = x1r + x3i;
        x0i = x1i + x3r;
        a[idx2] = wn4r * (x0r - x0i);
        a[idx2 + 1] = wn4r * (x0i + x0r);
        x0r = x1r - x3i;
        x0i = x1i - x3r;
        a[idx3] = -wn4r * (x0r + x0i);
        a[idx3 + 1] = -wn4r * (x0i - x0r);
        x0r = a[idx0 + 2] + a[idx2 + 2];
        x0i = -a[idx0 + 3] - a[idx2 + 3];
        x1r = a[idx0 + 2] - a[idx2 + 2];
        x1i = -a[idx0 + 3] + a[idx2 + 3];
        x2r = a[idx1 + 2] + a[idx3 + 2];
        x2i = a[idx1 + 3] + a[idx3 + 3];
        x3r = a[idx1 + 2] - a[idx3 + 2];
        x3i = a[idx1 + 3] - a[idx3 + 3];
        a[idx0 + 2] = x0r + x2r;
        a[idx0 + 3] = x0i - x2i;
        a[idx1 + 2] = x0r - x2r;
        a[idx1 + 3] = x0i + x2i;
        x0r = x1r + x3i;
        x0i = x1i + x3r;
        a[idx2 + 2] = wk1i * x0r - wk1r * x0i;
        a[idx2 + 3] = wk1i * x0i + wk1r * x0r;
        x0r = x1r - x3i;
        x0i = x1i - x3r;
        a[idx3 + 2] = wk3i * x0r + wk3r * x0i;
        a[idx3 + 3] = wk3i * x0i - wk3r * x0r;
    }

    public static void cftb1st(long n, DoubleLargeArray a, long offa, DoubleLargeArray w, long startw)
    {
        long j0, j1, j2, j3, k, m, mh;
        double wn4r, csc1, csc3, wk1r, wk1i, wk3r, wk3i, wd1r, wd1i, wd3r, wd3i;
        double x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        long idx0, idx1, idx2, idx3, idx4, idx5;
        mh = n >> 3l;
        m = 2 * mh;
        j1 = m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;

        x0r = a.getDouble(offa) + a.getDouble(idx2);
        x0i = -a.getDouble(offa + 1) - a.getDouble(idx2 + 1);
        x1r = a.getDouble(offa) - a.getDouble(idx2);
        x1i = -a.getDouble(offa + 1) + a.getDouble(idx2 + 1);
        x2r = a.getDouble(idx1) + a.getDouble(idx3);
        x2i = a.getDouble(idx1 + 1) + a.getDouble(idx3 + 1);
        x3r = a.getDouble(idx1) - a.getDouble(idx3);
        x3i = a.getDouble(idx1 + 1) - a.getDouble(idx3 + 1);
        a.setDouble(offa, x0r + x2r);
        a.setDouble(offa + 1, x0i - x2i);
        a.setDouble(idx1, x0r - x2r);
        a.setDouble(idx1 + 1, x0i + x2i);
        a.setDouble(idx2, x1r + x3i);
        a.setDouble(idx2 + 1, x1i + x3r);
        a.setDouble(idx3, x1r - x3i);
        a.setDouble(idx3 + 1, x1i - x3r);
        wn4r = w.getDouble(startw + 1);
        csc1 = w.getDouble(startw + 2);
        csc3 = w.getDouble(startw + 3);
        wd1r = 1;
        wd1i = 0;
        wd3r = 1;
        wd3i = 0;
        k = 0;
        for (long j = 2; j < mh - 2; j += 4) {
            k += 4;
            idx4 = startw + k;
            wk1r = csc1 * (wd1r + w.getDouble(idx4));
            wk1i = csc1 * (wd1i + w.getDouble(idx4 + 1));
            wk3r = csc3 * (wd3r + w.getDouble(idx4 + 2));
            wk3i = csc3 * (wd3i + w.getDouble(idx4 + 3));
            wd1r = w.getDouble(idx4);
            wd1i = w.getDouble(idx4 + 1);
            wd3r = w.getDouble(idx4 + 2);
            wd3i = w.getDouble(idx4 + 3);
            j1 = j + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            idx5 = offa + j;
            x0r = a.getDouble(idx5) + a.getDouble(idx2);
            x0i = -a.getDouble(idx5 + 1) - a.getDouble(idx2 + 1);
            x1r = a.getDouble(idx5) - a.getDouble(offa + j2);
            x1i = -a.getDouble(idx5 + 1) + a.getDouble(idx2 + 1);
            y0r = a.getDouble(idx5 + 2) + a.getDouble(idx2 + 2);
            y0i = -a.getDouble(idx5 + 3) - a.getDouble(idx2 + 3);
            y1r = a.getDouble(idx5 + 2) - a.getDouble(idx2 + 2);
            y1i = -a.getDouble(idx5 + 3) + a.getDouble(idx2 + 3);
            x2r = a.getDouble(idx1) + a.getDouble(idx3);
            x2i = a.getDouble(idx1 + 1) + a.getDouble(idx3 + 1);
            x3r = a.getDouble(idx1) - a.getDouble(idx3);
            x3i = a.getDouble(idx1 + 1) - a.getDouble(idx3 + 1);
            y2r = a.getDouble(idx1 + 2) + a.getDouble(idx3 + 2);
            y2i = a.getDouble(idx1 + 3) + a.getDouble(idx3 + 3);
            y3r = a.getDouble(idx1 + 2) - a.getDouble(idx3 + 2);
            y3i = a.getDouble(idx1 + 3) - a.getDouble(idx3 + 3);
            a.setDouble(idx5, x0r + x2r);
            a.setDouble(idx5 + 1, x0i - x2i);
            a.setDouble(idx5 + 2, y0r + y2r);
            a.setDouble(idx5 + 3, y0i - y2i);
            a.setDouble(idx1, x0r - x2r);
            a.setDouble(idx1 + 1, x0i + x2i);
            a.setDouble(idx1 + 2, y0r - y2r);
            a.setDouble(idx1 + 3, y0i + y2i);
            x0r = x1r + x3i;
            x0i = x1i + x3r;
            a.setDouble(idx2, wk1r * x0r - wk1i * x0i);
            a.setDouble(idx2 + 1, wk1r * x0i + wk1i * x0r);
            x0r = y1r + y3i;
            x0i = y1i + y3r;
            a.setDouble(idx2 + 2, wd1r * x0r - wd1i * x0i);
            a.setDouble(idx2 + 3, wd1r * x0i + wd1i * x0r);
            x0r = x1r - x3i;
            x0i = x1i - x3r;
            a.setDouble(idx3, wk3r * x0r + wk3i * x0i);
            a.setDouble(idx3 + 1, wk3r * x0i - wk3i * x0r);
            x0r = y1r - y3i;
            x0i = y1i - y3r;
            a.setDouble(idx3 + 2, wd3r * x0r + wd3i * x0i);
            a.setDouble(idx3 + 3, wd3r * x0i - wd3i * x0r);
            j0 = m - j;
            j1 = j0 + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx0 = offa + j0;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            x0r = a.getDouble(idx0) + a.getDouble(idx2);
            x0i = -a.getDouble(idx0 + 1) - a.getDouble(idx2 + 1);
            x1r = a.getDouble(idx0) - a.getDouble(idx2);
            x1i = -a.getDouble(idx0 + 1) + a.getDouble(idx2 + 1);
            y0r = a.getDouble(idx0 - 2) + a.getDouble(idx2 - 2);
            y0i = -a.getDouble(idx0 - 1) - a.getDouble(idx2 - 1);
            y1r = a.getDouble(idx0 - 2) - a.getDouble(idx2 - 2);
            y1i = -a.getDouble(idx0 - 1) + a.getDouble(idx2 - 1);
            x2r = a.getDouble(idx1) + a.getDouble(idx3);
            x2i = a.getDouble(idx1 + 1) + a.getDouble(idx3 + 1);
            x3r = a.getDouble(idx1) - a.getDouble(idx3);
            x3i = a.getDouble(idx1 + 1) - a.getDouble(idx3 + 1);
            y2r = a.getDouble(idx1 - 2) + a.getDouble(idx3 - 2);
            y2i = a.getDouble(idx1 - 1) + a.getDouble(idx3 - 1);
            y3r = a.getDouble(idx1 - 2) - a.getDouble(idx3 - 2);
            y3i = a.getDouble(idx1 - 1) - a.getDouble(idx3 - 1);
            a.setDouble(idx0, x0r + x2r);
            a.setDouble(idx0 + 1, x0i - x2i);
            a.setDouble(idx0 - 2, y0r + y2r);
            a.setDouble(idx0 - 1, y0i - y2i);
            a.setDouble(idx1, x0r - x2r);
            a.setDouble(idx1 + 1, x0i + x2i);
            a.setDouble(idx1 - 2, y0r - y2r);
            a.setDouble(idx1 - 1, y0i + y2i);
            x0r = x1r + x3i;
            x0i = x1i + x3r;
            a.setDouble(idx2, wk1i * x0r - wk1r * x0i);
            a.setDouble(idx2 + 1, wk1i * x0i + wk1r * x0r);
            x0r = y1r + y3i;
            x0i = y1i + y3r;
            a.setDouble(idx2 - 2, wd1i * x0r - wd1r * x0i);
            a.setDouble(idx2 - 1, wd1i * x0i + wd1r * x0r);
            x0r = x1r - x3i;
            x0i = x1i - x3r;
            a.setDouble(idx3, wk3i * x0r + wk3r * x0i);
            a.setDouble(idx3 + 1, wk3i * x0i - wk3r * x0r);
            x0r = y1r - y3i;
            x0i = y1i - y3r;
            a.setDouble(idx3 - 2, wd3i * x0r + wd3r * x0i);
            a.setDouble(idx3 - 1, wd3i * x0i - wd3r * x0r);
        }
        wk1r = csc1 * (wd1r + wn4r);
        wk1i = csc1 * (wd1i + wn4r);
        wk3r = csc3 * (wd3r - wn4r);
        wk3i = csc3 * (wd3i - wn4r);
        j0 = mh;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx0 = offa + j0;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a.getDouble(idx0 - 2) + a.getDouble(idx2 - 2);
        x0i = -a.getDouble(idx0 - 1) - a.getDouble(idx2 - 1);
        x1r = a.getDouble(idx0 - 2) - a.getDouble(idx2 - 2);
        x1i = -a.getDouble(idx0 - 1) + a.getDouble(idx2 - 1);
        x2r = a.getDouble(idx1 - 2) + a.getDouble(idx3 - 2);
        x2i = a.getDouble(idx1 - 1) + a.getDouble(idx3 - 1);
        x3r = a.getDouble(idx1 - 2) - a.getDouble(idx3 - 2);
        x3i = a.getDouble(idx1 - 1) - a.getDouble(idx3 - 1);
        a.setDouble(idx0 - 2, x0r + x2r);
        a.setDouble(idx0 - 1, x0i - x2i);
        a.setDouble(idx1 - 2, x0r - x2r);
        a.setDouble(idx1 - 1, x0i + x2i);
        x0r = x1r + x3i;
        x0i = x1i + x3r;
        a.setDouble(idx2 - 2, wk1r * x0r - wk1i * x0i);
        a.setDouble(idx2 - 1, wk1r * x0i + wk1i * x0r);
        x0r = x1r - x3i;
        x0i = x1i - x3r;
        a.setDouble(idx3 - 2, wk3r * x0r + wk3i * x0i);
        a.setDouble(idx3 - 1, wk3r * x0i - wk3i * x0r);
        x0r = a.getDouble(idx0) + a.getDouble(idx2);
        x0i = -a.getDouble(idx0 + 1) - a.getDouble(idx2 + 1);
        x1r = a.getDouble(idx0) - a.getDouble(idx2);
        x1i = -a.getDouble(idx0 + 1) + a.getDouble(idx2 + 1);
        x2r = a.getDouble(idx1) + a.getDouble(idx3);
        x2i = a.getDouble(idx1 + 1) + a.getDouble(idx3 + 1);
        x3r = a.getDouble(idx1) - a.getDouble(idx3);
        x3i = a.getDouble(idx1 + 1) - a.getDouble(idx3 + 1);
        a.setDouble(idx0, x0r + x2r);
        a.setDouble(idx0 + 1, x0i - x2i);
        a.setDouble(idx1, x0r - x2r);
        a.setDouble(idx1 + 1, x0i + x2i);
        x0r = x1r + x3i;
        x0i = x1i + x3r;
        a.setDouble(idx2, wn4r * (x0r - x0i));
        a.setDouble(idx2 + 1, wn4r * (x0i + x0r));
        x0r = x1r - x3i;
        x0i = x1i - x3r;
        a.setDouble(idx3, -wn4r * (x0r + x0i));
        a.setDouble(idx3 + 1, -wn4r * (x0i - x0r));
        x0r = a.getDouble(idx0 + 2) + a.getDouble(idx2 + 2);
        x0i = -a.getDouble(idx0 + 3) - a.getDouble(idx2 + 3);
        x1r = a.getDouble(idx0 + 2) - a.getDouble(idx2 + 2);
        x1i = -a.getDouble(idx0 + 3) + a.getDouble(idx2 + 3);
        x2r = a.getDouble(idx1 + 2) + a.getDouble(idx3 + 2);
        x2i = a.getDouble(idx1 + 3) + a.getDouble(idx3 + 3);
        x3r = a.getDouble(idx1 + 2) - a.getDouble(idx3 + 2);
        x3i = a.getDouble(idx1 + 3) - a.getDouble(idx3 + 3);
        a.setDouble(idx0 + 2, x0r + x2r);
        a.setDouble(idx0 + 3, x0i - x2i);
        a.setDouble(idx1 + 2, x0r - x2r);
        a.setDouble(idx1 + 3, x0i + x2i);
        x0r = x1r + x3i;
        x0i = x1i + x3r;
        a.setDouble(idx2 + 2, wk1i * x0r - wk1r * x0i);
        a.setDouble(idx2 + 3, wk1i * x0i + wk1r * x0r);
        x0r = x1r - x3i;
        x0i = x1i - x3r;
        a.setDouble(idx3 + 2, wk3i * x0r + wk3r * x0i);
        a.setDouble(idx3 + 3, wk3i * x0i - wk3r * x0r);
    }

    public static void cftrec4_th(final int n, final double[] a, final int offa, final int nw, final double[] w)
    {
        int i;
        int idiv4, m, nthreads;
        int idx = 0;
        nthreads = 2;
        idiv4 = 0;
        m = n >> 1;
        if (n >= CommonUtils.getThreadsBeginN_1D_FFT_4Threads()) {
            nthreads = 4;
            idiv4 = 1;
            m >>= 1;
        }
        Future<?>[] futures = new Future[nthreads];
        final int mf = m;
        for (i = 0; i < nthreads; i++) {
            final int firstIdx = offa + i * m;
            if (i != idiv4) {
                futures[idx++] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        int isplt, j, k, m;
                        int idx1 = firstIdx + mf;
                        m = n;
                        while (m > 512) {
                            m >>= 2;
                            cftmdl1(m, a, idx1 - m, w, nw - (m >> 1));
                        }
                        cftleaf(m, 1, a, idx1 - m, nw, w);
                        k = 0;
                        int idx2 = firstIdx - m;
                        for (j = mf - m; j > 0; j -= m) {
                            k++;
                            isplt = cfttree(m, j, k, a, firstIdx, nw, w);
                            cftleaf(m, isplt, a, idx2 + j, nw, w);
                        }
                    }
                });
            } else {
                futures[idx++] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        int isplt, j, k, m;
                        int idx1 = firstIdx + mf;
                        k = 1;
                        m = n;
                        while (m > 512) {
                            m >>= 2;
                            k <<= 2;
                            cftmdl2(m, a, idx1 - m, w, nw - m);
                        }
                        cftleaf(m, 0, a, idx1 - m, nw, w);
                        k >>= 1;
                        int idx2 = firstIdx - m;
                        for (j = mf - m; j > 0; j -= m) {
                            k++;
                            isplt = cfttree(m, j, k, a, firstIdx, nw, w);
                            cftleaf(m, isplt, a, idx2 + j, nw, w);
                        }
                    }
                });
            }
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(CommonUtils.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(CommonUtils.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static void cftrec4_th(final long n, final DoubleLargeArray a, final long offa, final long nw, final DoubleLargeArray w)
    {
        int i, idx = 0;
        int idiv4, nthreads;
        long m;
        nthreads = 2;
        idiv4 = 0;
        m = n >> 1l;
        if (n >= CommonUtils.getThreadsBeginN_1D_FFT_4Threads()) {
            nthreads = 4;
            idiv4 = 1;
            m >>= 1l;
        }
        Future<?>[] futures = new Future[nthreads];
        final long mf = m;
        for (i = 0; i < nthreads; i++) {
            final long firstIdx = offa + i * m;
            if (i != idiv4) {
                futures[idx++] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        long isplt, j, k, m;
                        long idx1 = firstIdx + mf;
                        m = n;
                        while (m > 512) {
                            m >>= 2;
                            cftmdl1(m, a, idx1 - m, w, nw - (m >> 1l));
                        }
                        cftleaf(m, 1, a, idx1 - m, nw, w);
                        k = 0;
                        long idx2 = firstIdx - m;
                        for (j = mf - m; j > 0; j -= m) {
                            k++;
                            isplt = cfttree(m, j, k, a, firstIdx, nw, w);
                            cftleaf(m, isplt, a, idx2 + j, nw, w);
                        }
                    }
                });
            } else {
                futures[idx++] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        long isplt, j, k, m;
                        long idx1 = firstIdx + mf;
                        k = 1;
                        m = n;
                        while (m > 512) {
                            m >>= 2;
                            k <<= 2;
                            cftmdl2(m, a, idx1 - m, w, nw - m);
                        }
                        cftleaf(m, 0, a, idx1 - m, nw, w);
                        k >>= 1l;
                        long idx2 = firstIdx - m;
                        for (j = mf - m; j > 0; j -= m) {
                            k++;
                            isplt = cfttree(m, j, k, a, firstIdx, nw, w);
                            cftleaf(m, isplt, a, idx2 + j, nw, w);
                        }
                    }
                });
            }
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(CommonUtils.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(CommonUtils.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static void cftrec4(int n, double[] a, int offa, int nw, double[] w)
    {
        int isplt, j, k, m;

        m = n;
        int idx1 = offa + n;
        while (m > 512) {
            m >>= 2;
            cftmdl1(m, a, idx1 - m, w, nw - (m >> 1));
        }
        cftleaf(m, 1, a, idx1 - m, nw, w);
        k = 0;
        int idx2 = offa - m;
        for (j = n - m; j > 0; j -= m) {
            k++;
            isplt = cfttree(m, j, k, a, offa, nw, w);
            cftleaf(m, isplt, a, idx2 + j, nw, w);
        }
    }

    public static void cftrec4(long n, DoubleLargeArray a, long offa, long nw, DoubleLargeArray w)
    {
        long isplt, j, k, m;

        m = n;
        long idx1 = offa + n;
        while (m > 512) {
            m >>= 2;
            cftmdl1(m, a, idx1 - m, w, nw - (m >> 1l));
        }
        cftleaf(m, 1, a, idx1 - m, nw, w);
        k = 0;
        long idx2 = offa - m;
        for (j = n - m; j > 0; j -= m) {
            k++;
            isplt = cfttree(m, j, k, a, offa, nw, w);
            cftleaf(m, isplt, a, idx2 + j, nw, w);
        }
    }

    public static int cfttree(int n, int j, int k, double[] a, int offa, int nw, double[] w)
    {
        int i, isplt, m;
        int idx1 = offa - n;
        if ((k & 3) != 0) {
            isplt = k & 1;
            if (isplt != 0) {
                cftmdl1(n, a, idx1 + j, w, nw - (n >> 1));
            } else {
                cftmdl2(n, a, idx1 + j, w, nw - n);
            }
        } else {
            m = n;
            for (i = k; (i & 3) == 0; i >>= 2) {
                m <<= 2;
            }
            isplt = i & 1;
            int idx2 = offa + j;
            if (isplt != 0) {
                while (m > 128) {
                    cftmdl1(m, a, idx2 - m, w, nw - (m >> 1));
                    m >>= 2;
                }
            } else {
                while (m > 128) {
                    cftmdl2(m, a, idx2 - m, w, nw - m);
                    m >>= 2;
                }
            }
        }
        return isplt;
    }

    public static long cfttree(long n, long j, long k, DoubleLargeArray a, long offa, long nw, DoubleLargeArray w)
    {
        long i, isplt, m;
        long idx1 = offa - n;
        if ((k & 3) != 0) {
            isplt = k & 1;
            if (isplt != 0) {
                cftmdl1(n, a, idx1 + j, w, nw - (n >> 1l));
            } else {
                cftmdl2(n, a, idx1 + j, w, nw - n);
            }
        } else {
            m = n;
            for (i = k; (i & 3) == 0; i >>= 2l) {
                m <<= 2l;
            }
            isplt = i & 1;
            long idx2 = offa + j;
            if (isplt != 0) {
                while (m > 128) {
                    cftmdl1(m, a, idx2 - m, w, nw - (m >> 1l));
                    m >>= 2l;
                }
            } else {
                while (m > 128) {
                    cftmdl2(m, a, idx2 - m, w, nw - m);
                    m >>= 2l;
                }
            }
        }
        return isplt;
    }

    public static void cftleaf(int n, int isplt, double[] a, int offa, int nw, double[] w)
    {
        if (n == 512) {
            cftmdl1(128, a, offa, w, nw - 64);
            cftf161(a, offa, w, nw - 8);
            cftf162(a, offa + 32, w, nw - 32);
            cftf161(a, offa + 64, w, nw - 8);
            cftf161(a, offa + 96, w, nw - 8);
            cftmdl2(128, a, offa + 128, w, nw - 128);
            cftf161(a, offa + 128, w, nw - 8);
            cftf162(a, offa + 160, w, nw - 32);
            cftf161(a, offa + 192, w, nw - 8);
            cftf162(a, offa + 224, w, nw - 32);
            cftmdl1(128, a, offa + 256, w, nw - 64);
            cftf161(a, offa + 256, w, nw - 8);
            cftf162(a, offa + 288, w, nw - 32);
            cftf161(a, offa + 320, w, nw - 8);
            cftf161(a, offa + 352, w, nw - 8);
            if (isplt != 0) {
                cftmdl1(128, a, offa + 384, w, nw - 64);
                cftf161(a, offa + 480, w, nw - 8);
            } else {
                cftmdl2(128, a, offa + 384, w, nw - 128);
                cftf162(a, offa + 480, w, nw - 32);
            }
            cftf161(a, offa + 384, w, nw - 8);
            cftf162(a, offa + 416, w, nw - 32);
            cftf161(a, offa + 448, w, nw - 8);
        } else {
            cftmdl1(64, a, offa, w, nw - 32);
            cftf081(a, offa, w, nw - 8);
            cftf082(a, offa + 16, w, nw - 8);
            cftf081(a, offa + 32, w, nw - 8);
            cftf081(a, offa + 48, w, nw - 8);
            cftmdl2(64, a, offa + 64, w, nw - 64);
            cftf081(a, offa + 64, w, nw - 8);
            cftf082(a, offa + 80, w, nw - 8);
            cftf081(a, offa + 96, w, nw - 8);
            cftf082(a, offa + 112, w, nw - 8);
            cftmdl1(64, a, offa + 128, w, nw - 32);
            cftf081(a, offa + 128, w, nw - 8);
            cftf082(a, offa + 144, w, nw - 8);
            cftf081(a, offa + 160, w, nw - 8);
            cftf081(a, offa + 176, w, nw - 8);
            if (isplt != 0) {
                cftmdl1(64, a, offa + 192, w, nw - 32);
                cftf081(a, offa + 240, w, nw - 8);
            } else {
                cftmdl2(64, a, offa + 192, w, nw - 64);
                cftf082(a, offa + 240, w, nw - 8);
            }
            cftf081(a, offa + 192, w, nw - 8);
            cftf082(a, offa + 208, w, nw - 8);
            cftf081(a, offa + 224, w, nw - 8);
        }
    }

    public static void cftleaf(long n, long isplt, DoubleLargeArray a, long offa, long nw, DoubleLargeArray w)
    {
        if (n == 512) {
            cftmdl1(128, a, offa, w, nw - 64);
            cftf161(a, offa, w, nw - 8);
            cftf162(a, offa + 32, w, nw - 32);
            cftf161(a, offa + 64, w, nw - 8);
            cftf161(a, offa + 96, w, nw - 8);
            cftmdl2(128, a, offa + 128, w, nw - 128);
            cftf161(a, offa + 128, w, nw - 8);
            cftf162(a, offa + 160, w, nw - 32);
            cftf161(a, offa + 192, w, nw - 8);
            cftf162(a, offa + 224, w, nw - 32);
            cftmdl1(128, a, offa + 256, w, nw - 64);
            cftf161(a, offa + 256, w, nw - 8);
            cftf162(a, offa + 288, w, nw - 32);
            cftf161(a, offa + 320, w, nw - 8);
            cftf161(a, offa + 352, w, nw - 8);
            if (isplt != 0) {
                cftmdl1(128, a, offa + 384, w, nw - 64);
                cftf161(a, offa + 480, w, nw - 8);
            } else {
                cftmdl2(128, a, offa + 384, w, nw - 128);
                cftf162(a, offa + 480, w, nw - 32);
            }
            cftf161(a, offa + 384, w, nw - 8);
            cftf162(a, offa + 416, w, nw - 32);
            cftf161(a, offa + 448, w, nw - 8);
        } else {
            cftmdl1(64, a, offa, w, nw - 32);
            cftf081(a, offa, w, nw - 8);
            cftf082(a, offa + 16, w, nw - 8);
            cftf081(a, offa + 32, w, nw - 8);
            cftf081(a, offa + 48, w, nw - 8);
            cftmdl2(64, a, offa + 64, w, nw - 64);
            cftf081(a, offa + 64, w, nw - 8);
            cftf082(a, offa + 80, w, nw - 8);
            cftf081(a, offa + 96, w, nw - 8);
            cftf082(a, offa + 112, w, nw - 8);
            cftmdl1(64, a, offa + 128, w, nw - 32);
            cftf081(a, offa + 128, w, nw - 8);
            cftf082(a, offa + 144, w, nw - 8);
            cftf081(a, offa + 160, w, nw - 8);
            cftf081(a, offa + 176, w, nw - 8);
            if (isplt != 0) {
                cftmdl1(64, a, offa + 192, w, nw - 32);
                cftf081(a, offa + 240, w, nw - 8);
            } else {
                cftmdl2(64, a, offa + 192, w, nw - 64);
                cftf082(a, offa + 240, w, nw - 8);
            }
            cftf081(a, offa + 192, w, nw - 8);
            cftf082(a, offa + 208, w, nw - 8);
            cftf081(a, offa + 224, w, nw - 8);
        }
    }

    public static void cftmdl1(int n, double[] a, int offa, double[] w, int startw)
    {
        int j0, j1, j2, j3, k, m, mh;
        double wn4r, wk1r, wk1i, wk3r, wk3i;
        double x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;
        int idx0, idx1, idx2, idx3, idx4, idx5;

        mh = n >> 3;
        m = 2 * mh;
        j1 = m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a[offa] + a[idx2];
        x0i = a[offa + 1] + a[idx2 + 1];
        x1r = a[offa] - a[idx2];
        x1i = a[offa + 1] - a[idx2 + 1];
        x2r = a[idx1] + a[idx3];
        x2i = a[idx1 + 1] + a[idx3 + 1];
        x3r = a[idx1] - a[idx3];
        x3i = a[idx1 + 1] - a[idx3 + 1];
        a[offa] = x0r + x2r;
        a[offa + 1] = x0i + x2i;
        a[idx1] = x0r - x2r;
        a[idx1 + 1] = x0i - x2i;
        a[idx2] = x1r - x3i;
        a[idx2 + 1] = x1i + x3r;
        a[idx3] = x1r + x3i;
        a[idx3 + 1] = x1i - x3r;
        wn4r = w[startw + 1];
        k = 0;
        for (int j = 2; j < mh; j += 2) {
            k += 4;
            idx4 = startw + k;
            wk1r = w[idx4];
            wk1i = w[idx4 + 1];
            wk3r = w[idx4 + 2];
            wk3i = w[idx4 + 3];
            j1 = j + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            idx5 = offa + j;
            x0r = a[idx5] + a[idx2];
            x0i = a[idx5 + 1] + a[idx2 + 1];
            x1r = a[idx5] - a[idx2];
            x1i = a[idx5 + 1] - a[idx2 + 1];
            x2r = a[idx1] + a[idx3];
            x2i = a[idx1 + 1] + a[idx3 + 1];
            x3r = a[idx1] - a[idx3];
            x3i = a[idx1 + 1] - a[idx3 + 1];
            a[idx5] = x0r + x2r;
            a[idx5 + 1] = x0i + x2i;
            a[idx1] = x0r - x2r;
            a[idx1 + 1] = x0i - x2i;
            x0r = x1r - x3i;
            x0i = x1i + x3r;
            a[idx2] = wk1r * x0r - wk1i * x0i;
            a[idx2 + 1] = wk1r * x0i + wk1i * x0r;
            x0r = x1r + x3i;
            x0i = x1i - x3r;
            a[idx3] = wk3r * x0r + wk3i * x0i;
            a[idx3 + 1] = wk3r * x0i - wk3i * x0r;
            j0 = m - j;
            j1 = j0 + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx0 = offa + j0;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            x0r = a[idx0] + a[idx2];
            x0i = a[idx0 + 1] + a[idx2 + 1];
            x1r = a[idx0] - a[idx2];
            x1i = a[idx0 + 1] - a[idx2 + 1];
            x2r = a[idx1] + a[idx3];
            x2i = a[idx1 + 1] + a[idx3 + 1];
            x3r = a[idx1] - a[idx3];
            x3i = a[idx1 + 1] - a[idx3 + 1];
            a[idx0] = x0r + x2r;
            a[idx0 + 1] = x0i + x2i;
            a[idx1] = x0r - x2r;
            a[idx1 + 1] = x0i - x2i;
            x0r = x1r - x3i;
            x0i = x1i + x3r;
            a[idx2] = wk1i * x0r - wk1r * x0i;
            a[idx2 + 1] = wk1i * x0i + wk1r * x0r;
            x0r = x1r + x3i;
            x0i = x1i - x3r;
            a[idx3] = wk3i * x0r + wk3r * x0i;
            a[idx3 + 1] = wk3i * x0i - wk3r * x0r;
        }
        j0 = mh;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx0 = offa + j0;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a[idx0] + a[idx2];
        x0i = a[idx0 + 1] + a[idx2 + 1];
        x1r = a[idx0] - a[idx2];
        x1i = a[idx0 + 1] - a[idx2 + 1];
        x2r = a[idx1] + a[idx3];
        x2i = a[idx1 + 1] + a[idx3 + 1];
        x3r = a[idx1] - a[idx3];
        x3i = a[idx1 + 1] - a[idx3 + 1];
        a[idx0] = x0r + x2r;
        a[idx0 + 1] = x0i + x2i;
        a[idx1] = x0r - x2r;
        a[idx1 + 1] = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a[idx2] = wn4r * (x0r - x0i);
        a[idx2 + 1] = wn4r * (x0i + x0r);
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a[idx3] = -wn4r * (x0r + x0i);
        a[idx3 + 1] = -wn4r * (x0i - x0r);
    }

    public static void cftmdl1(long n, DoubleLargeArray a, long offa, DoubleLargeArray w, long startw)
    {
        long j0, j1, j2, j3, k, m, mh;
        double wn4r, wk1r, wk1i, wk3r, wk3i;
        double x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;
        long idx0, idx1, idx2, idx3, idx4, idx5;

        mh = n >> 3l;
        m = 2 * mh;
        j1 = m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a.getDouble(offa) + a.getDouble(idx2);
        x0i = a.getDouble(offa + 1) + a.getDouble(idx2 + 1);
        x1r = a.getDouble(offa) - a.getDouble(idx2);
        x1i = a.getDouble(offa + 1) - a.getDouble(idx2 + 1);
        x2r = a.getDouble(idx1) + a.getDouble(idx3);
        x2i = a.getDouble(idx1 + 1) + a.getDouble(idx3 + 1);
        x3r = a.getDouble(idx1) - a.getDouble(idx3);
        x3i = a.getDouble(idx1 + 1) - a.getDouble(idx3 + 1);
        a.setDouble(offa, x0r + x2r);
        a.setDouble(offa + 1, x0i + x2i);
        a.setDouble(idx1, x0r - x2r);
        a.setDouble(idx1 + 1, x0i - x2i);
        a.setDouble(idx2, x1r - x3i);
        a.setDouble(idx2 + 1, x1i + x3r);
        a.setDouble(idx3, x1r + x3i);
        a.setDouble(idx3 + 1, x1i - x3r);
        wn4r = w.getDouble(startw + 1);
        k = 0;
        for (long j = 2; j < mh; j += 2) {
            k += 4;
            idx4 = startw + k;
            wk1r = w.getDouble(idx4);
            wk1i = w.getDouble(idx4 + 1);
            wk3r = w.getDouble(idx4 + 2);
            wk3i = w.getDouble(idx4 + 3);
            j1 = j + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            idx5 = offa + j;
            x0r = a.getDouble(idx5) + a.getDouble(idx2);
            x0i = a.getDouble(idx5 + 1) + a.getDouble(idx2 + 1);
            x1r = a.getDouble(idx5) - a.getDouble(idx2);
            x1i = a.getDouble(idx5 + 1) - a.getDouble(idx2 + 1);
            x2r = a.getDouble(idx1) + a.getDouble(idx3);
            x2i = a.getDouble(idx1 + 1) + a.getDouble(idx3 + 1);
            x3r = a.getDouble(idx1) - a.getDouble(idx3);
            x3i = a.getDouble(idx1 + 1) - a.getDouble(idx3 + 1);
            a.setDouble(idx5, x0r + x2r);
            a.setDouble(idx5 + 1, x0i + x2i);
            a.setDouble(idx1, x0r - x2r);
            a.setDouble(idx1 + 1, x0i - x2i);
            x0r = x1r - x3i;
            x0i = x1i + x3r;
            a.setDouble(idx2, wk1r * x0r - wk1i * x0i);
            a.setDouble(idx2 + 1, wk1r * x0i + wk1i * x0r);
            x0r = x1r + x3i;
            x0i = x1i - x3r;
            a.setDouble(idx3, wk3r * x0r + wk3i * x0i);
            a.setDouble(idx3 + 1, wk3r * x0i - wk3i * x0r);
            j0 = m - j;
            j1 = j0 + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx0 = offa + j0;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            x0r = a.getDouble(idx0) + a.getDouble(idx2);
            x0i = a.getDouble(idx0 + 1) + a.getDouble(idx2 + 1);
            x1r = a.getDouble(idx0) - a.getDouble(idx2);
            x1i = a.getDouble(idx0 + 1) - a.getDouble(idx2 + 1);
            x2r = a.getDouble(idx1) + a.getDouble(idx3);
            x2i = a.getDouble(idx1 + 1) + a.getDouble(idx3 + 1);
            x3r = a.getDouble(idx1) - a.getDouble(idx3);
            x3i = a.getDouble(idx1 + 1) - a.getDouble(idx3 + 1);
            a.setDouble(idx0, x0r + x2r);
            a.setDouble(idx0 + 1, x0i + x2i);
            a.setDouble(idx1, x0r - x2r);
            a.setDouble(idx1 + 1, x0i - x2i);
            x0r = x1r - x3i;
            x0i = x1i + x3r;
            a.setDouble(idx2, wk1i * x0r - wk1r * x0i);
            a.setDouble(idx2 + 1, wk1i * x0i + wk1r * x0r);
            x0r = x1r + x3i;
            x0i = x1i - x3r;
            a.setDouble(idx3, wk3i * x0r + wk3r * x0i);
            a.setDouble(idx3 + 1, wk3i * x0i - wk3r * x0r);
        }
        j0 = mh;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx0 = offa + j0;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a.getDouble(idx0) + a.getDouble(idx2);
        x0i = a.getDouble(idx0 + 1) + a.getDouble(idx2 + 1);
        x1r = a.getDouble(idx0) - a.getDouble(idx2);
        x1i = a.getDouble(idx0 + 1) - a.getDouble(idx2 + 1);
        x2r = a.getDouble(idx1) + a.getDouble(idx3);
        x2i = a.getDouble(idx1 + 1) + a.getDouble(idx3 + 1);
        x3r = a.getDouble(idx1) - a.getDouble(idx3);
        x3i = a.getDouble(idx1 + 1) - a.getDouble(idx3 + 1);
        a.setDouble(idx0, x0r + x2r);
        a.setDouble(idx0 + 1, x0i + x2i);
        a.setDouble(idx1, x0r - x2r);
        a.setDouble(idx1 + 1, x0i - x2i);
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a.setDouble(idx2, wn4r * (x0r - x0i));
        a.setDouble(idx2 + 1, wn4r * (x0i + x0r));
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a.setDouble(idx3, -wn4r * (x0r + x0i));
        a.setDouble(idx3 + 1, -wn4r * (x0i - x0r));
    }

    public static void cftmdl2(int n, double[] a, int offa, double[] w, int startw)
    {
        int j0, j1, j2, j3, k, kr, m, mh;
        double wn4r, wk1r, wk1i, wk3r, wk3i, wd1r, wd1i, wd3r, wd3i;
        double x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, y0r, y0i, y2r, y2i;
        int idx0, idx1, idx2, idx3, idx4, idx5, idx6;

        mh = n >> 3;
        m = 2 * mh;
        wn4r = w[startw + 1];
        j1 = m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a[offa] - a[idx2 + 1];
        x0i = a[offa + 1] + a[idx2];
        x1r = a[offa] + a[idx2 + 1];
        x1i = a[offa + 1] - a[idx2];
        x2r = a[idx1] - a[idx3 + 1];
        x2i = a[idx1 + 1] + a[idx3];
        x3r = a[idx1] + a[idx3 + 1];
        x3i = a[idx1 + 1] - a[idx3];
        y0r = wn4r * (x2r - x2i);
        y0i = wn4r * (x2i + x2r);
        a[offa] = x0r + y0r;
        a[offa + 1] = x0i + y0i;
        a[idx1] = x0r - y0r;
        a[idx1 + 1] = x0i - y0i;
        y0r = wn4r * (x3r - x3i);
        y0i = wn4r * (x3i + x3r);
        a[idx2] = x1r - y0i;
        a[idx2 + 1] = x1i + y0r;
        a[idx3] = x1r + y0i;
        a[idx3 + 1] = x1i - y0r;
        k = 0;
        kr = 2 * m;
        for (int j = 2; j < mh; j += 2) {
            k += 4;
            idx4 = startw + k;
            wk1r = w[idx4];
            wk1i = w[idx4 + 1];
            wk3r = w[idx4 + 2];
            wk3i = w[idx4 + 3];
            kr -= 4;
            idx5 = startw + kr;
            wd1i = w[idx5];
            wd1r = w[idx5 + 1];
            wd3i = w[idx5 + 2];
            wd3r = w[idx5 + 3];
            j1 = j + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            idx6 = offa + j;
            x0r = a[idx6] - a[idx2 + 1];
            x0i = a[idx6 + 1] + a[idx2];
            x1r = a[idx6] + a[idx2 + 1];
            x1i = a[idx6 + 1] - a[idx2];
            x2r = a[idx1] - a[idx3 + 1];
            x2i = a[idx1 + 1] + a[idx3];
            x3r = a[idx1] + a[idx3 + 1];
            x3i = a[idx1 + 1] - a[idx3];
            y0r = wk1r * x0r - wk1i * x0i;
            y0i = wk1r * x0i + wk1i * x0r;
            y2r = wd1r * x2r - wd1i * x2i;
            y2i = wd1r * x2i + wd1i * x2r;
            a[idx6] = y0r + y2r;
            a[idx6 + 1] = y0i + y2i;
            a[idx1] = y0r - y2r;
            a[idx1 + 1] = y0i - y2i;
            y0r = wk3r * x1r + wk3i * x1i;
            y0i = wk3r * x1i - wk3i * x1r;
            y2r = wd3r * x3r + wd3i * x3i;
            y2i = wd3r * x3i - wd3i * x3r;
            a[idx2] = y0r + y2r;
            a[idx2 + 1] = y0i + y2i;
            a[idx3] = y0r - y2r;
            a[idx3 + 1] = y0i - y2i;
            j0 = m - j;
            j1 = j0 + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx0 = offa + j0;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            x0r = a[idx0] - a[idx2 + 1];
            x0i = a[idx0 + 1] + a[idx2];
            x1r = a[idx0] + a[idx2 + 1];
            x1i = a[idx0 + 1] - a[idx2];
            x2r = a[idx1] - a[idx3 + 1];
            x2i = a[idx1 + 1] + a[idx3];
            x3r = a[idx1] + a[idx3 + 1];
            x3i = a[idx1 + 1] - a[idx3];
            y0r = wd1i * x0r - wd1r * x0i;
            y0i = wd1i * x0i + wd1r * x0r;
            y2r = wk1i * x2r - wk1r * x2i;
            y2i = wk1i * x2i + wk1r * x2r;
            a[idx0] = y0r + y2r;
            a[idx0 + 1] = y0i + y2i;
            a[idx1] = y0r - y2r;
            a[idx1 + 1] = y0i - y2i;
            y0r = wd3i * x1r + wd3r * x1i;
            y0i = wd3i * x1i - wd3r * x1r;
            y2r = wk3i * x3r + wk3r * x3i;
            y2i = wk3i * x3i - wk3r * x3r;
            a[idx2] = y0r + y2r;
            a[idx2 + 1] = y0i + y2i;
            a[idx3] = y0r - y2r;
            a[idx3 + 1] = y0i - y2i;
        }
        wk1r = w[startw + m];
        wk1i = w[startw + m + 1];
        j0 = mh;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx0 = offa + j0;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a[idx0] - a[idx2 + 1];
        x0i = a[idx0 + 1] + a[idx2];
        x1r = a[idx0] + a[idx2 + 1];
        x1i = a[idx0 + 1] - a[idx2];
        x2r = a[idx1] - a[idx3 + 1];
        x2i = a[idx1 + 1] + a[idx3];
        x3r = a[idx1] + a[idx3 + 1];
        x3i = a[idx1 + 1] - a[idx3];
        y0r = wk1r * x0r - wk1i * x0i;
        y0i = wk1r * x0i + wk1i * x0r;
        y2r = wk1i * x2r - wk1r * x2i;
        y2i = wk1i * x2i + wk1r * x2r;
        a[idx0] = y0r + y2r;
        a[idx0 + 1] = y0i + y2i;
        a[idx1] = y0r - y2r;
        a[idx1 + 1] = y0i - y2i;
        y0r = wk1i * x1r - wk1r * x1i;
        y0i = wk1i * x1i + wk1r * x1r;
        y2r = wk1r * x3r - wk1i * x3i;
        y2i = wk1r * x3i + wk1i * x3r;
        a[idx2] = y0r - y2r;
        a[idx2 + 1] = y0i - y2i;
        a[idx3] = y0r + y2r;
        a[idx3 + 1] = y0i + y2i;
    }

    public static void cftmdl2(long n, DoubleLargeArray a, long offa, DoubleLargeArray w, long startw)
    {
        long j0, j1, j2, j3, k, kr, m, mh;
        double wn4r, wk1r, wk1i, wk3r, wk3i, wd1r, wd1i, wd3r, wd3i;
        double x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, y0r, y0i, y2r, y2i;
        long idx0, idx1, idx2, idx3, idx4, idx5, idx6;

        mh = n >> 3l;
        m = 2 * mh;
        wn4r = w.getDouble(startw + 1);
        j1 = m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a.getDouble(offa) - a.getDouble(idx2 + 1);
        x0i = a.getDouble(offa + 1) + a.getDouble(idx2);
        x1r = a.getDouble(offa) + a.getDouble(idx2 + 1);
        x1i = a.getDouble(offa + 1) - a.getDouble(idx2);
        x2r = a.getDouble(idx1) - a.getDouble(idx3 + 1);
        x2i = a.getDouble(idx1 + 1) + a.getDouble(idx3);
        x3r = a.getDouble(idx1) + a.getDouble(idx3 + 1);
        x3i = a.getDouble(idx1 + 1) - a.getDouble(idx3);
        y0r = wn4r * (x2r - x2i);
        y0i = wn4r * (x2i + x2r);
        a.setDouble(offa, x0r + y0r);
        a.setDouble(offa + 1, x0i + y0i);
        a.setDouble(idx1, x0r - y0r);
        a.setDouble(idx1 + 1, x0i - y0i);
        y0r = wn4r * (x3r - x3i);
        y0i = wn4r * (x3i + x3r);
        a.setDouble(idx2, x1r - y0i);
        a.setDouble(idx2 + 1, x1i + y0r);
        a.setDouble(idx3, x1r + y0i);
        a.setDouble(idx3 + 1, x1i - y0r);
        k = 0;
        kr = 2 * m;
        for (int j = 2; j < mh; j += 2) {
            k += 4;
            idx4 = startw + k;
            wk1r = w.getDouble(idx4);
            wk1i = w.getDouble(idx4 + 1);
            wk3r = w.getDouble(idx4 + 2);
            wk3i = w.getDouble(idx4 + 3);
            kr -= 4;
            idx5 = startw + kr;
            wd1i = w.getDouble(idx5);
            wd1r = w.getDouble(idx5 + 1);
            wd3i = w.getDouble(idx5 + 2);
            wd3r = w.getDouble(idx5 + 3);
            j1 = j + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            idx6 = offa + j;
            x0r = a.getDouble(idx6) - a.getDouble(idx2 + 1);
            x0i = a.getDouble(idx6 + 1) + a.getDouble(idx2);
            x1r = a.getDouble(idx6) + a.getDouble(idx2 + 1);
            x1i = a.getDouble(idx6 + 1) - a.getDouble(idx2);
            x2r = a.getDouble(idx1) - a.getDouble(idx3 + 1);
            x2i = a.getDouble(idx1 + 1) + a.getDouble(idx3);
            x3r = a.getDouble(idx1) + a.getDouble(idx3 + 1);
            x3i = a.getDouble(idx1 + 1) - a.getDouble(idx3);
            y0r = wk1r * x0r - wk1i * x0i;
            y0i = wk1r * x0i + wk1i * x0r;
            y2r = wd1r * x2r - wd1i * x2i;
            y2i = wd1r * x2i + wd1i * x2r;
            a.setDouble(idx6, y0r + y2r);
            a.setDouble(idx6 + 1, y0i + y2i);
            a.setDouble(idx1, y0r - y2r);
            a.setDouble(idx1 + 1, y0i - y2i);
            y0r = wk3r * x1r + wk3i * x1i;
            y0i = wk3r * x1i - wk3i * x1r;
            y2r = wd3r * x3r + wd3i * x3i;
            y2i = wd3r * x3i - wd3i * x3r;
            a.setDouble(idx2, y0r + y2r);
            a.setDouble(idx2 + 1, y0i + y2i);
            a.setDouble(idx3, y0r - y2r);
            a.setDouble(idx3 + 1, y0i - y2i);
            j0 = m - j;
            j1 = j0 + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx0 = offa + j0;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            x0r = a.getDouble(idx0) - a.getDouble(idx2 + 1);
            x0i = a.getDouble(idx0 + 1) + a.getDouble(idx2);
            x1r = a.getDouble(idx0) + a.getDouble(idx2 + 1);
            x1i = a.getDouble(idx0 + 1) - a.getDouble(idx2);
            x2r = a.getDouble(idx1) - a.getDouble(idx3 + 1);
            x2i = a.getDouble(idx1 + 1) + a.getDouble(idx3);
            x3r = a.getDouble(idx1) + a.getDouble(idx3 + 1);
            x3i = a.getDouble(idx1 + 1) - a.getDouble(idx3);
            y0r = wd1i * x0r - wd1r * x0i;
            y0i = wd1i * x0i + wd1r * x0r;
            y2r = wk1i * x2r - wk1r * x2i;
            y2i = wk1i * x2i + wk1r * x2r;
            a.setDouble(idx0, y0r + y2r);
            a.setDouble(idx0 + 1, y0i + y2i);
            a.setDouble(idx1, y0r - y2r);
            a.setDouble(idx1 + 1, y0i - y2i);
            y0r = wd3i * x1r + wd3r * x1i;
            y0i = wd3i * x1i - wd3r * x1r;
            y2r = wk3i * x3r + wk3r * x3i;
            y2i = wk3i * x3i - wk3r * x3r;
            a.setDouble(idx2, y0r + y2r);
            a.setDouble(idx2 + 1, y0i + y2i);
            a.setDouble(idx3, y0r - y2r);
            a.setDouble(idx3 + 1, y0i - y2i);
        }
        wk1r = w.getDouble(startw + m);
        wk1i = w.getDouble(startw + m + 1);
        j0 = mh;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx0 = offa + j0;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a.getDouble(idx0) - a.getDouble(idx2 + 1);
        x0i = a.getDouble(idx0 + 1) + a.getDouble(idx2);
        x1r = a.getDouble(idx0) + a.getDouble(idx2 + 1);
        x1i = a.getDouble(idx0 + 1) - a.getDouble(idx2);
        x2r = a.getDouble(idx1) - a.getDouble(idx3 + 1);
        x2i = a.getDouble(idx1 + 1) + a.getDouble(idx3);
        x3r = a.getDouble(idx1) + a.getDouble(idx3 + 1);
        x3i = a.getDouble(idx1 + 1) - a.getDouble(idx3);
        y0r = wk1r * x0r - wk1i * x0i;
        y0i = wk1r * x0i + wk1i * x0r;
        y2r = wk1i * x2r - wk1r * x2i;
        y2i = wk1i * x2i + wk1r * x2r;
        a.setDouble(idx0, y0r + y2r);
        a.setDouble(idx0 + 1, y0i + y2i);
        a.setDouble(idx1, y0r - y2r);
        a.setDouble(idx1 + 1, y0i - y2i);
        y0r = wk1i * x1r - wk1r * x1i;
        y0i = wk1i * x1i + wk1r * x1r;
        y2r = wk1r * x3r - wk1i * x3i;
        y2i = wk1r * x3i + wk1i * x3r;
        a.setDouble(idx2, y0r - y2r);
        a.setDouble(idx2 + 1, y0i - y2i);
        a.setDouble(idx3, y0r + y2r);
        a.setDouble(idx3 + 1, y0i + y2i);
    }

    public static void cftfx41(int n, double[] a, int offa, int nw, double[] w)
    {
        if (n == 128) {
            cftf161(a, offa, w, nw - 8);
            cftf162(a, offa + 32, w, nw - 32);
            cftf161(a, offa + 64, w, nw - 8);
            cftf161(a, offa + 96, w, nw - 8);
        } else {
            cftf081(a, offa, w, nw - 8);
            cftf082(a, offa + 16, w, nw - 8);
            cftf081(a, offa + 32, w, nw - 8);
            cftf081(a, offa + 48, w, nw - 8);
        }
    }

    public static void cftfx41(long n, DoubleLargeArray a, long offa, long nw, DoubleLargeArray w)
    {
        if (n == 128) {
            cftf161(a, offa, w, nw - 8);
            cftf162(a, offa + 32, w, nw - 32);
            cftf161(a, offa + 64, w, nw - 8);
            cftf161(a, offa + 96, w, nw - 8);
        } else {
            cftf081(a, offa, w, nw - 8);
            cftf082(a, offa + 16, w, nw - 8);
            cftf081(a, offa + 32, w, nw - 8);
            cftf081(a, offa + 48, w, nw - 8);
        }
    }

    public static void cftf161(double[] a, int offa, double[] w, int startw)
    {
        double wn4r, wk1r, wk1i, x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i, y8r, y8i, y9r, y9i, y10r, y10i, y11r, y11i, y12r, y12i, y13r, y13i, y14r, y14i, y15r, y15i;

        wn4r = w[startw + 1];
        wk1r = w[startw + 2];
        wk1i = w[startw + 3];

        x0r = a[offa] + a[offa + 16];
        x0i = a[offa + 1] + a[offa + 17];
        x1r = a[offa] - a[offa + 16];
        x1i = a[offa + 1] - a[offa + 17];
        x2r = a[offa + 8] + a[offa + 24];
        x2i = a[offa + 9] + a[offa + 25];
        x3r = a[offa + 8] - a[offa + 24];
        x3i = a[offa + 9] - a[offa + 25];
        y0r = x0r + x2r;
        y0i = x0i + x2i;
        y4r = x0r - x2r;
        y4i = x0i - x2i;
        y8r = x1r - x3i;
        y8i = x1i + x3r;
        y12r = x1r + x3i;
        y12i = x1i - x3r;
        x0r = a[offa + 2] + a[offa + 18];
        x0i = a[offa + 3] + a[offa + 19];
        x1r = a[offa + 2] - a[offa + 18];
        x1i = a[offa + 3] - a[offa + 19];
        x2r = a[offa + 10] + a[offa + 26];
        x2i = a[offa + 11] + a[offa + 27];
        x3r = a[offa + 10] - a[offa + 26];
        x3i = a[offa + 11] - a[offa + 27];
        y1r = x0r + x2r;
        y1i = x0i + x2i;
        y5r = x0r - x2r;
        y5i = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        y9r = wk1r * x0r - wk1i * x0i;
        y9i = wk1r * x0i + wk1i * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        y13r = wk1i * x0r - wk1r * x0i;
        y13i = wk1i * x0i + wk1r * x0r;
        x0r = a[offa + 4] + a[offa + 20];
        x0i = a[offa + 5] + a[offa + 21];
        x1r = a[offa + 4] - a[offa + 20];
        x1i = a[offa + 5] - a[offa + 21];
        x2r = a[offa + 12] + a[offa + 28];
        x2i = a[offa + 13] + a[offa + 29];
        x3r = a[offa + 12] - a[offa + 28];
        x3i = a[offa + 13] - a[offa + 29];
        y2r = x0r + x2r;
        y2i = x0i + x2i;
        y6r = x0r - x2r;
        y6i = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        y10r = wn4r * (x0r - x0i);
        y10i = wn4r * (x0i + x0r);
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        y14r = wn4r * (x0r + x0i);
        y14i = wn4r * (x0i - x0r);
        x0r = a[offa + 6] + a[offa + 22];
        x0i = a[offa + 7] + a[offa + 23];
        x1r = a[offa + 6] - a[offa + 22];
        x1i = a[offa + 7] - a[offa + 23];
        x2r = a[offa + 14] + a[offa + 30];
        x2i = a[offa + 15] + a[offa + 31];
        x3r = a[offa + 14] - a[offa + 30];
        x3i = a[offa + 15] - a[offa + 31];
        y3r = x0r + x2r;
        y3i = x0i + x2i;
        y7r = x0r - x2r;
        y7i = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        y11r = wk1i * x0r - wk1r * x0i;
        y11i = wk1i * x0i + wk1r * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        y15r = wk1r * x0r - wk1i * x0i;
        y15i = wk1r * x0i + wk1i * x0r;
        x0r = y12r - y14r;
        x0i = y12i - y14i;
        x1r = y12r + y14r;
        x1i = y12i + y14i;
        x2r = y13r - y15r;
        x2i = y13i - y15i;
        x3r = y13r + y15r;
        x3i = y13i + y15i;
        a[offa + 24] = x0r + x2r;
        a[offa + 25] = x0i + x2i;
        a[offa + 26] = x0r - x2r;
        a[offa + 27] = x0i - x2i;
        a[offa + 28] = x1r - x3i;
        a[offa + 29] = x1i + x3r;
        a[offa + 30] = x1r + x3i;
        a[offa + 31] = x1i - x3r;
        x0r = y8r + y10r;
        x0i = y8i + y10i;
        x1r = y8r - y10r;
        x1i = y8i - y10i;
        x2r = y9r + y11r;
        x2i = y9i + y11i;
        x3r = y9r - y11r;
        x3i = y9i - y11i;
        a[offa + 16] = x0r + x2r;
        a[offa + 17] = x0i + x2i;
        a[offa + 18] = x0r - x2r;
        a[offa + 19] = x0i - x2i;
        a[offa + 20] = x1r - x3i;
        a[offa + 21] = x1i + x3r;
        a[offa + 22] = x1r + x3i;
        a[offa + 23] = x1i - x3r;
        x0r = y5r - y7i;
        x0i = y5i + y7r;
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        x0r = y5r + y7i;
        x0i = y5i - y7r;
        x3r = wn4r * (x0r - x0i);
        x3i = wn4r * (x0i + x0r);
        x0r = y4r - y6i;
        x0i = y4i + y6r;
        x1r = y4r + y6i;
        x1i = y4i - y6r;
        a[offa + 8] = x0r + x2r;
        a[offa + 9] = x0i + x2i;
        a[offa + 10] = x0r - x2r;
        a[offa + 11] = x0i - x2i;
        a[offa + 12] = x1r - x3i;
        a[offa + 13] = x1i + x3r;
        a[offa + 14] = x1r + x3i;
        a[offa + 15] = x1i - x3r;
        x0r = y0r + y2r;
        x0i = y0i + y2i;
        x1r = y0r - y2r;
        x1i = y0i - y2i;
        x2r = y1r + y3r;
        x2i = y1i + y3i;
        x3r = y1r - y3r;
        x3i = y1i - y3i;
        a[offa] = x0r + x2r;
        a[offa + 1] = x0i + x2i;
        a[offa + 2] = x0r - x2r;
        a[offa + 3] = x0i - x2i;
        a[offa + 4] = x1r - x3i;
        a[offa + 5] = x1i + x3r;
        a[offa + 6] = x1r + x3i;
        a[offa + 7] = x1i - x3r;
    }

    public static void cftf161(DoubleLargeArray a, long offa, DoubleLargeArray w, long startw)
    {
        double wn4r, wk1r, wk1i, x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i, y8r, y8i, y9r, y9i, y10r, y10i, y11r, y11i, y12r, y12i, y13r, y13i, y14r, y14i, y15r, y15i;

        wn4r = w.getDouble(startw + 1);
        wk1r = w.getDouble(startw + 2);
        wk1i = w.getDouble(startw + 3);

        x0r = a.getDouble(offa) + a.getDouble(offa + 16);
        x0i = a.getDouble(offa + 1) + a.getDouble(offa + 17);
        x1r = a.getDouble(offa) - a.getDouble(offa + 16);
        x1i = a.getDouble(offa + 1) - a.getDouble(offa + 17);
        x2r = a.getDouble(offa + 8) + a.getDouble(offa + 24);
        x2i = a.getDouble(offa + 9) + a.getDouble(offa + 25);
        x3r = a.getDouble(offa + 8) - a.getDouble(offa + 24);
        x3i = a.getDouble(offa + 9) - a.getDouble(offa + 25);
        y0r = x0r + x2r;
        y0i = x0i + x2i;
        y4r = x0r - x2r;
        y4i = x0i - x2i;
        y8r = x1r - x3i;
        y8i = x1i + x3r;
        y12r = x1r + x3i;
        y12i = x1i - x3r;
        x0r = a.getDouble(offa + 2) + a.getDouble(offa + 18);
        x0i = a.getDouble(offa + 3) + a.getDouble(offa + 19);
        x1r = a.getDouble(offa + 2) - a.getDouble(offa + 18);
        x1i = a.getDouble(offa + 3) - a.getDouble(offa + 19);
        x2r = a.getDouble(offa + 10) + a.getDouble(offa + 26);
        x2i = a.getDouble(offa + 11) + a.getDouble(offa + 27);
        x3r = a.getDouble(offa + 10) - a.getDouble(offa + 26);
        x3i = a.getDouble(offa + 11) - a.getDouble(offa + 27);
        y1r = x0r + x2r;
        y1i = x0i + x2i;
        y5r = x0r - x2r;
        y5i = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        y9r = wk1r * x0r - wk1i * x0i;
        y9i = wk1r * x0i + wk1i * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        y13r = wk1i * x0r - wk1r * x0i;
        y13i = wk1i * x0i + wk1r * x0r;
        x0r = a.getDouble(offa + 4) + a.getDouble(offa + 20);
        x0i = a.getDouble(offa + 5) + a.getDouble(offa + 21);
        x1r = a.getDouble(offa + 4) - a.getDouble(offa + 20);
        x1i = a.getDouble(offa + 5) - a.getDouble(offa + 21);
        x2r = a.getDouble(offa + 12) + a.getDouble(offa + 28);
        x2i = a.getDouble(offa + 13) + a.getDouble(offa + 29);
        x3r = a.getDouble(offa + 12) - a.getDouble(offa + 28);
        x3i = a.getDouble(offa + 13) - a.getDouble(offa + 29);
        y2r = x0r + x2r;
        y2i = x0i + x2i;
        y6r = x0r - x2r;
        y6i = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        y10r = wn4r * (x0r - x0i);
        y10i = wn4r * (x0i + x0r);
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        y14r = wn4r * (x0r + x0i);
        y14i = wn4r * (x0i - x0r);
        x0r = a.getDouble(offa + 6) + a.getDouble(offa + 22);
        x0i = a.getDouble(offa + 7) + a.getDouble(offa + 23);
        x1r = a.getDouble(offa + 6) - a.getDouble(offa + 22);
        x1i = a.getDouble(offa + 7) - a.getDouble(offa + 23);
        x2r = a.getDouble(offa + 14) + a.getDouble(offa + 30);
        x2i = a.getDouble(offa + 15) + a.getDouble(offa + 31);
        x3r = a.getDouble(offa + 14) - a.getDouble(offa + 30);
        x3i = a.getDouble(offa + 15) - a.getDouble(offa + 31);
        y3r = x0r + x2r;
        y3i = x0i + x2i;
        y7r = x0r - x2r;
        y7i = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        y11r = wk1i * x0r - wk1r * x0i;
        y11i = wk1i * x0i + wk1r * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        y15r = wk1r * x0r - wk1i * x0i;
        y15i = wk1r * x0i + wk1i * x0r;
        x0r = y12r - y14r;
        x0i = y12i - y14i;
        x1r = y12r + y14r;
        x1i = y12i + y14i;
        x2r = y13r - y15r;
        x2i = y13i - y15i;
        x3r = y13r + y15r;
        x3i = y13i + y15i;
        a.setDouble(offa + 24, x0r + x2r);
        a.setDouble(offa + 25, x0i + x2i);
        a.setDouble(offa + 26, x0r - x2r);
        a.setDouble(offa + 27, x0i - x2i);
        a.setDouble(offa + 28, x1r - x3i);
        a.setDouble(offa + 29, x1i + x3r);
        a.setDouble(offa + 30, x1r + x3i);
        a.setDouble(offa + 31, x1i - x3r);
        x0r = y8r + y10r;
        x0i = y8i + y10i;
        x1r = y8r - y10r;
        x1i = y8i - y10i;
        x2r = y9r + y11r;
        x2i = y9i + y11i;
        x3r = y9r - y11r;
        x3i = y9i - y11i;
        a.setDouble(offa + 16, x0r + x2r);
        a.setDouble(offa + 17, x0i + x2i);
        a.setDouble(offa + 18, x0r - x2r);
        a.setDouble(offa + 19, x0i - x2i);
        a.setDouble(offa + 20, x1r - x3i);
        a.setDouble(offa + 21, x1i + x3r);
        a.setDouble(offa + 22, x1r + x3i);
        a.setDouble(offa + 23, x1i - x3r);
        x0r = y5r - y7i;
        x0i = y5i + y7r;
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        x0r = y5r + y7i;
        x0i = y5i - y7r;
        x3r = wn4r * (x0r - x0i);
        x3i = wn4r * (x0i + x0r);
        x0r = y4r - y6i;
        x0i = y4i + y6r;
        x1r = y4r + y6i;
        x1i = y4i - y6r;
        a.setDouble(offa + 8, x0r + x2r);
        a.setDouble(offa + 9, x0i + x2i);
        a.setDouble(offa + 10, x0r - x2r);
        a.setDouble(offa + 11, x0i - x2i);
        a.setDouble(offa + 12, x1r - x3i);
        a.setDouble(offa + 13, x1i + x3r);
        a.setDouble(offa + 14, x1r + x3i);
        a.setDouble(offa + 15, x1i - x3r);
        x0r = y0r + y2r;
        x0i = y0i + y2i;
        x1r = y0r - y2r;
        x1i = y0i - y2i;
        x2r = y1r + y3r;
        x2i = y1i + y3i;
        x3r = y1r - y3r;
        x3i = y1i - y3i;
        a.setDouble(offa, x0r + x2r);
        a.setDouble(offa + 1, x0i + x2i);
        a.setDouble(offa + 2, x0r - x2r);
        a.setDouble(offa + 3, x0i - x2i);
        a.setDouble(offa + 4, x1r - x3i);
        a.setDouble(offa + 5, x1i + x3r);
        a.setDouble(offa + 6, x1r + x3i);
        a.setDouble(offa + 7, x1i - x3r);
    }

    public static void cftf162(double[] a, int offa, double[] w, int startw)
    {
        double wn4r, wk1r, wk1i, wk2r, wk2i, wk3r, wk3i, x0r, x0i, x1r, x1i, x2r, x2i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i, y8r, y8i, y9r, y9i, y10r, y10i, y11r, y11i, y12r, y12i, y13r, y13i, y14r, y14i, y15r, y15i;

        wn4r = w[startw + 1];
        wk1r = w[startw + 4];
        wk1i = w[startw + 5];
        wk3r = w[startw + 6];
        wk3i = -w[startw + 7];
        wk2r = w[startw + 8];
        wk2i = w[startw + 9];
        x1r = a[offa] - a[offa + 17];
        x1i = a[offa + 1] + a[offa + 16];
        x0r = a[offa + 8] - a[offa + 25];
        x0i = a[offa + 9] + a[offa + 24];
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        y0r = x1r + x2r;
        y0i = x1i + x2i;
        y4r = x1r - x2r;
        y4i = x1i - x2i;
        x1r = a[offa] + a[offa + 17];
        x1i = a[offa + 1] - a[offa + 16];
        x0r = a[offa + 8] + a[offa + 25];
        x0i = a[offa + 9] - a[offa + 24];
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        y8r = x1r - x2i;
        y8i = x1i + x2r;
        y12r = x1r + x2i;
        y12i = x1i - x2r;
        x0r = a[offa + 2] - a[offa + 19];
        x0i = a[offa + 3] + a[offa + 18];
        x1r = wk1r * x0r - wk1i * x0i;
        x1i = wk1r * x0i + wk1i * x0r;
        x0r = a[offa + 10] - a[offa + 27];
        x0i = a[offa + 11] + a[offa + 26];
        x2r = wk3i * x0r - wk3r * x0i;
        x2i = wk3i * x0i + wk3r * x0r;
        y1r = x1r + x2r;
        y1i = x1i + x2i;
        y5r = x1r - x2r;
        y5i = x1i - x2i;
        x0r = a[offa + 2] + a[offa + 19];
        x0i = a[offa + 3] - a[offa + 18];
        x1r = wk3r * x0r - wk3i * x0i;
        x1i = wk3r * x0i + wk3i * x0r;
        x0r = a[offa + 10] + a[offa + 27];
        x0i = a[offa + 11] - a[offa + 26];
        x2r = wk1r * x0r + wk1i * x0i;
        x2i = wk1r * x0i - wk1i * x0r;
        y9r = x1r - x2r;
        y9i = x1i - x2i;
        y13r = x1r + x2r;
        y13i = x1i + x2i;
        x0r = a[offa + 4] - a[offa + 21];
        x0i = a[offa + 5] + a[offa + 20];
        x1r = wk2r * x0r - wk2i * x0i;
        x1i = wk2r * x0i + wk2i * x0r;
        x0r = a[offa + 12] - a[offa + 29];
        x0i = a[offa + 13] + a[offa + 28];
        x2r = wk2i * x0r - wk2r * x0i;
        x2i = wk2i * x0i + wk2r * x0r;
        y2r = x1r + x2r;
        y2i = x1i + x2i;
        y6r = x1r - x2r;
        y6i = x1i - x2i;
        x0r = a[offa + 4] + a[offa + 21];
        x0i = a[offa + 5] - a[offa + 20];
        x1r = wk2i * x0r - wk2r * x0i;
        x1i = wk2i * x0i + wk2r * x0r;
        x0r = a[offa + 12] + a[offa + 29];
        x0i = a[offa + 13] - a[offa + 28];
        x2r = wk2r * x0r - wk2i * x0i;
        x2i = wk2r * x0i + wk2i * x0r;
        y10r = x1r - x2r;
        y10i = x1i - x2i;
        y14r = x1r + x2r;
        y14i = x1i + x2i;
        x0r = a[offa + 6] - a[offa + 23];
        x0i = a[offa + 7] + a[offa + 22];
        x1r = wk3r * x0r - wk3i * x0i;
        x1i = wk3r * x0i + wk3i * x0r;
        x0r = a[offa + 14] - a[offa + 31];
        x0i = a[offa + 15] + a[offa + 30];
        x2r = wk1i * x0r - wk1r * x0i;
        x2i = wk1i * x0i + wk1r * x0r;
        y3r = x1r + x2r;
        y3i = x1i + x2i;
        y7r = x1r - x2r;
        y7i = x1i - x2i;
        x0r = a[offa + 6] + a[offa + 23];
        x0i = a[offa + 7] - a[offa + 22];
        x1r = wk1i * x0r + wk1r * x0i;
        x1i = wk1i * x0i - wk1r * x0r;
        x0r = a[offa + 14] + a[offa + 31];
        x0i = a[offa + 15] - a[offa + 30];
        x2r = wk3i * x0r - wk3r * x0i;
        x2i = wk3i * x0i + wk3r * x0r;
        y11r = x1r + x2r;
        y11i = x1i + x2i;
        y15r = x1r - x2r;
        y15i = x1i - x2i;
        x1r = y0r + y2r;
        x1i = y0i + y2i;
        x2r = y1r + y3r;
        x2i = y1i + y3i;
        a[offa] = x1r + x2r;
        a[offa + 1] = x1i + x2i;
        a[offa + 2] = x1r - x2r;
        a[offa + 3] = x1i - x2i;
        x1r = y0r - y2r;
        x1i = y0i - y2i;
        x2r = y1r - y3r;
        x2i = y1i - y3i;
        a[offa + 4] = x1r - x2i;
        a[offa + 5] = x1i + x2r;
        a[offa + 6] = x1r + x2i;
        a[offa + 7] = x1i - x2r;
        x1r = y4r - y6i;
        x1i = y4i + y6r;
        x0r = y5r - y7i;
        x0i = y5i + y7r;
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        a[offa + 8] = x1r + x2r;
        a[offa + 9] = x1i + x2i;
        a[offa + 10] = x1r - x2r;
        a[offa + 11] = x1i - x2i;
        x1r = y4r + y6i;
        x1i = y4i - y6r;
        x0r = y5r + y7i;
        x0i = y5i - y7r;
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        a[offa + 12] = x1r - x2i;
        a[offa + 13] = x1i + x2r;
        a[offa + 14] = x1r + x2i;
        a[offa + 15] = x1i - x2r;
        x1r = y8r + y10r;
        x1i = y8i + y10i;
        x2r = y9r - y11r;
        x2i = y9i - y11i;
        a[offa + 16] = x1r + x2r;
        a[offa + 17] = x1i + x2i;
        a[offa + 18] = x1r - x2r;
        a[offa + 19] = x1i - x2i;
        x1r = y8r - y10r;
        x1i = y8i - y10i;
        x2r = y9r + y11r;
        x2i = y9i + y11i;
        a[offa + 20] = x1r - x2i;
        a[offa + 21] = x1i + x2r;
        a[offa + 22] = x1r + x2i;
        a[offa + 23] = x1i - x2r;
        x1r = y12r - y14i;
        x1i = y12i + y14r;
        x0r = y13r + y15i;
        x0i = y13i - y15r;
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        a[offa + 24] = x1r + x2r;
        a[offa + 25] = x1i + x2i;
        a[offa + 26] = x1r - x2r;
        a[offa + 27] = x1i - x2i;
        x1r = y12r + y14i;
        x1i = y12i - y14r;
        x0r = y13r - y15i;
        x0i = y13i + y15r;
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        a[offa + 28] = x1r - x2i;
        a[offa + 29] = x1i + x2r;
        a[offa + 30] = x1r + x2i;
        a[offa + 31] = x1i - x2r;
    }

    public static void cftf162(DoubleLargeArray a, long offa, DoubleLargeArray w, long startw)
    {
        double wn4r, wk1r, wk1i, wk2r, wk2i, wk3r, wk3i, x0r, x0i, x1r, x1i, x2r, x2i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i, y8r, y8i, y9r, y9i, y10r, y10i, y11r, y11i, y12r, y12i, y13r, y13i, y14r, y14i, y15r, y15i;

        wn4r = w.getDouble(startw + 1);
        wk1r = w.getDouble(startw + 4);
        wk1i = w.getDouble(startw + 5);
        wk3r = w.getDouble(startw + 6);
        wk3i = -w.getDouble(startw + 7);
        wk2r = w.getDouble(startw + 8);
        wk2i = w.getDouble(startw + 9);
        x1r = a.getDouble(offa) - a.getDouble(offa + 17);
        x1i = a.getDouble(offa + 1) + a.getDouble(offa + 16);
        x0r = a.getDouble(offa + 8) - a.getDouble(offa + 25);
        x0i = a.getDouble(offa + 9) + a.getDouble(offa + 24);
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        y0r = x1r + x2r;
        y0i = x1i + x2i;
        y4r = x1r - x2r;
        y4i = x1i - x2i;
        x1r = a.getDouble(offa) + a.getDouble(offa + 17);
        x1i = a.getDouble(offa + 1) - a.getDouble(offa + 16);
        x0r = a.getDouble(offa + 8) + a.getDouble(offa + 25);
        x0i = a.getDouble(offa + 9) - a.getDouble(offa + 24);
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        y8r = x1r - x2i;
        y8i = x1i + x2r;
        y12r = x1r + x2i;
        y12i = x1i - x2r;
        x0r = a.getDouble(offa + 2) - a.getDouble(offa + 19);
        x0i = a.getDouble(offa + 3) + a.getDouble(offa + 18);
        x1r = wk1r * x0r - wk1i * x0i;
        x1i = wk1r * x0i + wk1i * x0r;
        x0r = a.getDouble(offa + 10) - a.getDouble(offa + 27);
        x0i = a.getDouble(offa + 11) + a.getDouble(offa + 26);
        x2r = wk3i * x0r - wk3r * x0i;
        x2i = wk3i * x0i + wk3r * x0r;
        y1r = x1r + x2r;
        y1i = x1i + x2i;
        y5r = x1r - x2r;
        y5i = x1i - x2i;
        x0r = a.getDouble(offa + 2) + a.getDouble(offa + 19);
        x0i = a.getDouble(offa + 3) - a.getDouble(offa + 18);
        x1r = wk3r * x0r - wk3i * x0i;
        x1i = wk3r * x0i + wk3i * x0r;
        x0r = a.getDouble(offa + 10) + a.getDouble(offa + 27);
        x0i = a.getDouble(offa + 11) - a.getDouble(offa + 26);
        x2r = wk1r * x0r + wk1i * x0i;
        x2i = wk1r * x0i - wk1i * x0r;
        y9r = x1r - x2r;
        y9i = x1i - x2i;
        y13r = x1r + x2r;
        y13i = x1i + x2i;
        x0r = a.getDouble(offa + 4) - a.getDouble(offa + 21);
        x0i = a.getDouble(offa + 5) + a.getDouble(offa + 20);
        x1r = wk2r * x0r - wk2i * x0i;
        x1i = wk2r * x0i + wk2i * x0r;
        x0r = a.getDouble(offa + 12) - a.getDouble(offa + 29);
        x0i = a.getDouble(offa + 13) + a.getDouble(offa + 28);
        x2r = wk2i * x0r - wk2r * x0i;
        x2i = wk2i * x0i + wk2r * x0r;
        y2r = x1r + x2r;
        y2i = x1i + x2i;
        y6r = x1r - x2r;
        y6i = x1i - x2i;
        x0r = a.getDouble(offa + 4) + a.getDouble(offa + 21);
        x0i = a.getDouble(offa + 5) - a.getDouble(offa + 20);
        x1r = wk2i * x0r - wk2r * x0i;
        x1i = wk2i * x0i + wk2r * x0r;
        x0r = a.getDouble(offa + 12) + a.getDouble(offa + 29);
        x0i = a.getDouble(offa + 13) - a.getDouble(offa + 28);
        x2r = wk2r * x0r - wk2i * x0i;
        x2i = wk2r * x0i + wk2i * x0r;
        y10r = x1r - x2r;
        y10i = x1i - x2i;
        y14r = x1r + x2r;
        y14i = x1i + x2i;
        x0r = a.getDouble(offa + 6) - a.getDouble(offa + 23);
        x0i = a.getDouble(offa + 7) + a.getDouble(offa + 22);
        x1r = wk3r * x0r - wk3i * x0i;
        x1i = wk3r * x0i + wk3i * x0r;
        x0r = a.getDouble(offa + 14) - a.getDouble(offa + 31);
        x0i = a.getDouble(offa + 15) + a.getDouble(offa + 30);
        x2r = wk1i * x0r - wk1r * x0i;
        x2i = wk1i * x0i + wk1r * x0r;
        y3r = x1r + x2r;
        y3i = x1i + x2i;
        y7r = x1r - x2r;
        y7i = x1i - x2i;
        x0r = a.getDouble(offa + 6) + a.getDouble(offa + 23);
        x0i = a.getDouble(offa + 7) - a.getDouble(offa + 22);
        x1r = wk1i * x0r + wk1r * x0i;
        x1i = wk1i * x0i - wk1r * x0r;
        x0r = a.getDouble(offa + 14) + a.getDouble(offa + 31);
        x0i = a.getDouble(offa + 15) - a.getDouble(offa + 30);
        x2r = wk3i * x0r - wk3r * x0i;
        x2i = wk3i * x0i + wk3r * x0r;
        y11r = x1r + x2r;
        y11i = x1i + x2i;
        y15r = x1r - x2r;
        y15i = x1i - x2i;
        x1r = y0r + y2r;
        x1i = y0i + y2i;
        x2r = y1r + y3r;
        x2i = y1i + y3i;
        a.setDouble(offa, x1r + x2r);
        a.setDouble(offa + 1, x1i + x2i);
        a.setDouble(offa + 2, x1r - x2r);
        a.setDouble(offa + 3, x1i - x2i);
        x1r = y0r - y2r;
        x1i = y0i - y2i;
        x2r = y1r - y3r;
        x2i = y1i - y3i;
        a.setDouble(offa + 4, x1r - x2i);
        a.setDouble(offa + 5, x1i + x2r);
        a.setDouble(offa + 6, x1r + x2i);
        a.setDouble(offa + 7, x1i - x2r);
        x1r = y4r - y6i;
        x1i = y4i + y6r;
        x0r = y5r - y7i;
        x0i = y5i + y7r;
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        a.setDouble(offa + 8, x1r + x2r);
        a.setDouble(offa + 9, x1i + x2i);
        a.setDouble(offa + 10, x1r - x2r);
        a.setDouble(offa + 11, x1i - x2i);
        x1r = y4r + y6i;
        x1i = y4i - y6r;
        x0r = y5r + y7i;
        x0i = y5i - y7r;
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        a.setDouble(offa + 12, x1r - x2i);
        a.setDouble(offa + 13, x1i + x2r);
        a.setDouble(offa + 14, x1r + x2i);
        a.setDouble(offa + 15, x1i - x2r);
        x1r = y8r + y10r;
        x1i = y8i + y10i;
        x2r = y9r - y11r;
        x2i = y9i - y11i;
        a.setDouble(offa + 16, x1r + x2r);
        a.setDouble(offa + 17, x1i + x2i);
        a.setDouble(offa + 18, x1r - x2r);
        a.setDouble(offa + 19, x1i - x2i);
        x1r = y8r - y10r;
        x1i = y8i - y10i;
        x2r = y9r + y11r;
        x2i = y9i + y11i;
        a.setDouble(offa + 20, x1r - x2i);
        a.setDouble(offa + 21, x1i + x2r);
        a.setDouble(offa + 22, x1r + x2i);
        a.setDouble(offa + 23, x1i - x2r);
        x1r = y12r - y14i;
        x1i = y12i + y14r;
        x0r = y13r + y15i;
        x0i = y13i - y15r;
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        a.setDouble(offa + 24, x1r + x2r);
        a.setDouble(offa + 25, x1i + x2i);
        a.setDouble(offa + 26, x1r - x2r);
        a.setDouble(offa + 27, x1i - x2i);
        x1r = y12r + y14i;
        x1i = y12i - y14r;
        x0r = y13r - y15i;
        x0i = y13i + y15r;
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        a.setDouble(offa + 28, x1r - x2i);
        a.setDouble(offa + 29, x1i + x2r);
        a.setDouble(offa + 30, x1r + x2i);
        a.setDouble(offa + 31, x1i - x2r);
    }

    public static void cftf081(double[] a, int offa, double[] w, int startw)
    {
        double wn4r, x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;

        wn4r = w[startw + 1];
        x0r = a[offa] + a[offa + 8];
        x0i = a[offa + 1] + a[offa + 9];
        x1r = a[offa] - a[offa + 8];
        x1i = a[offa + 1] - a[offa + 9];
        x2r = a[offa + 4] + a[offa + 12];
        x2i = a[offa + 5] + a[offa + 13];
        x3r = a[offa + 4] - a[offa + 12];
        x3i = a[offa + 5] - a[offa + 13];
        y0r = x0r + x2r;
        y0i = x0i + x2i;
        y2r = x0r - x2r;
        y2i = x0i - x2i;
        y1r = x1r - x3i;
        y1i = x1i + x3r;
        y3r = x1r + x3i;
        y3i = x1i - x3r;
        x0r = a[offa + 2] + a[offa + 10];
        x0i = a[offa + 3] + a[offa + 11];
        x1r = a[offa + 2] - a[offa + 10];
        x1i = a[offa + 3] - a[offa + 11];
        x2r = a[offa + 6] + a[offa + 14];
        x2i = a[offa + 7] + a[offa + 15];
        x3r = a[offa + 6] - a[offa + 14];
        x3i = a[offa + 7] - a[offa + 15];
        y4r = x0r + x2r;
        y4i = x0i + x2i;
        y6r = x0r - x2r;
        y6i = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        x2r = x1r + x3i;
        x2i = x1i - x3r;
        y5r = wn4r * (x0r - x0i);
        y5i = wn4r * (x0r + x0i);
        y7r = wn4r * (x2r - x2i);
        y7i = wn4r * (x2r + x2i);
        a[offa + 8] = y1r + y5r;
        a[offa + 9] = y1i + y5i;
        a[offa + 10] = y1r - y5r;
        a[offa + 11] = y1i - y5i;
        a[offa + 12] = y3r - y7i;
        a[offa + 13] = y3i + y7r;
        a[offa + 14] = y3r + y7i;
        a[offa + 15] = y3i - y7r;
        a[offa] = y0r + y4r;
        a[offa + 1] = y0i + y4i;
        a[offa + 2] = y0r - y4r;
        a[offa + 3] = y0i - y4i;
        a[offa + 4] = y2r - y6i;
        a[offa + 5] = y2i + y6r;
        a[offa + 6] = y2r + y6i;
        a[offa + 7] = y2i - y6r;
    }

    public static void cftf081(DoubleLargeArray a, long offa, DoubleLargeArray w, long startw)
    {
        double wn4r, x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;

        wn4r = w.getDouble(startw + 1);
        x0r = a.getDouble(offa) + a.getDouble(offa + 8);
        x0i = a.getDouble(offa + 1) + a.getDouble(offa + 9);
        x1r = a.getDouble(offa) - a.getDouble(offa + 8);
        x1i = a.getDouble(offa + 1) - a.getDouble(offa + 9);
        x2r = a.getDouble(offa + 4) + a.getDouble(offa + 12);
        x2i = a.getDouble(offa + 5) + a.getDouble(offa + 13);
        x3r = a.getDouble(offa + 4) - a.getDouble(offa + 12);
        x3i = a.getDouble(offa + 5) - a.getDouble(offa + 13);
        y0r = x0r + x2r;
        y0i = x0i + x2i;
        y2r = x0r - x2r;
        y2i = x0i - x2i;
        y1r = x1r - x3i;
        y1i = x1i + x3r;
        y3r = x1r + x3i;
        y3i = x1i - x3r;
        x0r = a.getDouble(offa + 2) + a.getDouble(offa + 10);
        x0i = a.getDouble(offa + 3) + a.getDouble(offa + 11);
        x1r = a.getDouble(offa + 2) - a.getDouble(offa + 10);
        x1i = a.getDouble(offa + 3) - a.getDouble(offa + 11);
        x2r = a.getDouble(offa + 6) + a.getDouble(offa + 14);
        x2i = a.getDouble(offa + 7) + a.getDouble(offa + 15);
        x3r = a.getDouble(offa + 6) - a.getDouble(offa + 14);
        x3i = a.getDouble(offa + 7) - a.getDouble(offa + 15);
        y4r = x0r + x2r;
        y4i = x0i + x2i;
        y6r = x0r - x2r;
        y6i = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        x2r = x1r + x3i;
        x2i = x1i - x3r;
        y5r = wn4r * (x0r - x0i);
        y5i = wn4r * (x0r + x0i);
        y7r = wn4r * (x2r - x2i);
        y7i = wn4r * (x2r + x2i);
        a.setDouble(offa + 8, y1r + y5r);
        a.setDouble(offa + 9, y1i + y5i);
        a.setDouble(offa + 10, y1r - y5r);
        a.setDouble(offa + 11, y1i - y5i);
        a.setDouble(offa + 12, y3r - y7i);
        a.setDouble(offa + 13, y3i + y7r);
        a.setDouble(offa + 14, y3r + y7i);
        a.setDouble(offa + 15, y3i - y7r);
        a.setDouble(offa, y0r + y4r);
        a.setDouble(offa + 1, y0i + y4i);
        a.setDouble(offa + 2, y0r - y4r);
        a.setDouble(offa + 3, y0i - y4i);
        a.setDouble(offa + 4, y2r - y6i);
        a.setDouble(offa + 5, y2i + y6r);
        a.setDouble(offa + 6, y2r + y6i);
        a.setDouble(offa + 7, y2i - y6r);
    }

    public static void cftf082(double[] a, int offa, double[] w, int startw)
    {
        double wn4r, wk1r, wk1i, x0r, x0i, x1r, x1i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;

        wn4r = w[startw + 1];
        wk1r = w[startw + 2];
        wk1i = w[startw + 3];
        y0r = a[offa] - a[offa + 9];
        y0i = a[offa + 1] + a[offa + 8];
        y1r = a[offa] + a[offa + 9];
        y1i = a[offa + 1] - a[offa + 8];
        x0r = a[offa + 4] - a[offa + 13];
        x0i = a[offa + 5] + a[offa + 12];
        y2r = wn4r * (x0r - x0i);
        y2i = wn4r * (x0i + x0r);
        x0r = a[offa + 4] + a[offa + 13];
        x0i = a[offa + 5] - a[offa + 12];
        y3r = wn4r * (x0r - x0i);
        y3i = wn4r * (x0i + x0r);
        x0r = a[offa + 2] - a[offa + 11];
        x0i = a[offa + 3] + a[offa + 10];
        y4r = wk1r * x0r - wk1i * x0i;
        y4i = wk1r * x0i + wk1i * x0r;
        x0r = a[offa + 2] + a[offa + 11];
        x0i = a[offa + 3] - a[offa + 10];
        y5r = wk1i * x0r - wk1r * x0i;
        y5i = wk1i * x0i + wk1r * x0r;
        x0r = a[offa + 6] - a[offa + 15];
        x0i = a[offa + 7] + a[offa + 14];
        y6r = wk1i * x0r - wk1r * x0i;
        y6i = wk1i * x0i + wk1r * x0r;
        x0r = a[offa + 6] + a[offa + 15];
        x0i = a[offa + 7] - a[offa + 14];
        y7r = wk1r * x0r - wk1i * x0i;
        y7i = wk1r * x0i + wk1i * x0r;
        x0r = y0r + y2r;
        x0i = y0i + y2i;
        x1r = y4r + y6r;
        x1i = y4i + y6i;
        a[offa] = x0r + x1r;
        a[offa + 1] = x0i + x1i;
        a[offa + 2] = x0r - x1r;
        a[offa + 3] = x0i - x1i;
        x0r = y0r - y2r;
        x0i = y0i - y2i;
        x1r = y4r - y6r;
        x1i = y4i - y6i;
        a[offa + 4] = x0r - x1i;
        a[offa + 5] = x0i + x1r;
        a[offa + 6] = x0r + x1i;
        a[offa + 7] = x0i - x1r;
        x0r = y1r - y3i;
        x0i = y1i + y3r;
        x1r = y5r - y7r;
        x1i = y5i - y7i;
        a[offa + 8] = x0r + x1r;
        a[offa + 9] = x0i + x1i;
        a[offa + 10] = x0r - x1r;
        a[offa + 11] = x0i - x1i;
        x0r = y1r + y3i;
        x0i = y1i - y3r;
        x1r = y5r + y7r;
        x1i = y5i + y7i;
        a[offa + 12] = x0r - x1i;
        a[offa + 13] = x0i + x1r;
        a[offa + 14] = x0r + x1i;
        a[offa + 15] = x0i - x1r;
    }

    public static void cftf082(DoubleLargeArray a, long offa, DoubleLargeArray w, long startw)
    {
        double wn4r, wk1r, wk1i, x0r, x0i, x1r, x1i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;

        wn4r = w.getDouble(startw + 1);
        wk1r = w.getDouble(startw + 2);
        wk1i = w.getDouble(startw + 3);
        y0r = a.getDouble(offa) - a.getDouble(offa + 9);
        y0i = a.getDouble(offa + 1) + a.getDouble(offa + 8);
        y1r = a.getDouble(offa) + a.getDouble(offa + 9);
        y1i = a.getDouble(offa + 1) - a.getDouble(offa + 8);
        x0r = a.getDouble(offa + 4) - a.getDouble(offa + 13);
        x0i = a.getDouble(offa + 5) + a.getDouble(offa + 12);
        y2r = wn4r * (x0r - x0i);
        y2i = wn4r * (x0i + x0r);
        x0r = a.getDouble(offa + 4) + a.getDouble(offa + 13);
        x0i = a.getDouble(offa + 5) - a.getDouble(offa + 12);
        y3r = wn4r * (x0r - x0i);
        y3i = wn4r * (x0i + x0r);
        x0r = a.getDouble(offa + 2) - a.getDouble(offa + 11);
        x0i = a.getDouble(offa + 3) + a.getDouble(offa + 10);
        y4r = wk1r * x0r - wk1i * x0i;
        y4i = wk1r * x0i + wk1i * x0r;
        x0r = a.getDouble(offa + 2) + a.getDouble(offa + 11);
        x0i = a.getDouble(offa + 3) - a.getDouble(offa + 10);
        y5r = wk1i * x0r - wk1r * x0i;
        y5i = wk1i * x0i + wk1r * x0r;
        x0r = a.getDouble(offa + 6) - a.getDouble(offa + 15);
        x0i = a.getDouble(offa + 7) + a.getDouble(offa + 14);
        y6r = wk1i * x0r - wk1r * x0i;
        y6i = wk1i * x0i + wk1r * x0r;
        x0r = a.getDouble(offa + 6) + a.getDouble(offa + 15);
        x0i = a.getDouble(offa + 7) - a.getDouble(offa + 14);
        y7r = wk1r * x0r - wk1i * x0i;
        y7i = wk1r * x0i + wk1i * x0r;
        x0r = y0r + y2r;
        x0i = y0i + y2i;
        x1r = y4r + y6r;
        x1i = y4i + y6i;
        a.setDouble(offa, x0r + x1r);
        a.setDouble(offa + 1, x0i + x1i);
        a.setDouble(offa + 2, x0r - x1r);
        a.setDouble(offa + 3, x0i - x1i);
        x0r = y0r - y2r;
        x0i = y0i - y2i;
        x1r = y4r - y6r;
        x1i = y4i - y6i;
        a.setDouble(offa + 4, x0r - x1i);
        a.setDouble(offa + 5, x0i + x1r);
        a.setDouble(offa + 6, x0r + x1i);
        a.setDouble(offa + 7, x0i - x1r);
        x0r = y1r - y3i;
        x0i = y1i + y3r;
        x1r = y5r - y7r;
        x1i = y5i - y7i;
        a.setDouble(offa + 8, x0r + x1r);
        a.setDouble(offa + 9, x0i + x1i);
        a.setDouble(offa + 10, x0r - x1r);
        a.setDouble(offa + 11, x0i - x1i);
        x0r = y1r + y3i;
        x0i = y1i - y3r;
        x1r = y5r + y7r;
        x1i = y5i + y7i;
        a.setDouble(offa + 12, x0r - x1i);
        a.setDouble(offa + 13, x0i + x1r);
        a.setDouble(offa + 14, x0r + x1i);
        a.setDouble(offa + 15, x0i - x1r);
    }

    public static void cftf040(double[] a, int offa)
    {
        double x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;

        x0r = a[offa] + a[offa + 4];
        x0i = a[offa + 1] + a[offa + 5];
        x1r = a[offa] - a[offa + 4];
        x1i = a[offa + 1] - a[offa + 5];
        x2r = a[offa + 2] + a[offa + 6];
        x2i = a[offa + 3] + a[offa + 7];
        x3r = a[offa + 2] - a[offa + 6];
        x3i = a[offa + 3] - a[offa + 7];
        a[offa] = x0r + x2r;
        a[offa + 1] = x0i + x2i;
        a[offa + 2] = x1r - x3i;
        a[offa + 3] = x1i + x3r;
        a[offa + 4] = x0r - x2r;
        a[offa + 5] = x0i - x2i;
        a[offa + 6] = x1r + x3i;
        a[offa + 7] = x1i - x3r;
    }

    public static void cftf040(DoubleLargeArray a, long offa)
    {
        double x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;

        x0r = a.getDouble(offa) + a.getDouble(offa + 4);
        x0i = a.getDouble(offa + 1) + a.getDouble(offa + 5);
        x1r = a.getDouble(offa) - a.getDouble(offa + 4);
        x1i = a.getDouble(offa + 1) - a.getDouble(offa + 5);
        x2r = a.getDouble(offa + 2) + a.getDouble(offa + 6);
        x2i = a.getDouble(offa + 3) + a.getDouble(offa + 7);
        x3r = a.getDouble(offa + 2) - a.getDouble(offa + 6);
        x3i = a.getDouble(offa + 3) - a.getDouble(offa + 7);
        a.setDouble(offa, x0r + x2r);
        a.setDouble(offa + 1, x0i + x2i);
        a.setDouble(offa + 2, x1r - x3i);
        a.setDouble(offa + 3, x1i + x3r);
        a.setDouble(offa + 4, x0r - x2r);
        a.setDouble(offa + 5, x0i - x2i);
        a.setDouble(offa + 6, x1r + x3i);
        a.setDouble(offa + 7, x1i - x3r);
    }

    public static void cftb040(double[] a, int offa)
    {
        double x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;

        x0r = a[offa] + a[offa + 4];
        x0i = a[offa + 1] + a[offa + 5];
        x1r = a[offa] - a[offa + 4];
        x1i = a[offa + 1] - a[offa + 5];
        x2r = a[offa + 2] + a[offa + 6];
        x2i = a[offa + 3] + a[offa + 7];
        x3r = a[offa + 2] - a[offa + 6];
        x3i = a[offa + 3] - a[offa + 7];
        a[offa] = x0r + x2r;
        a[offa + 1] = x0i + x2i;
        a[offa + 2] = x1r + x3i;
        a[offa + 3] = x1i - x3r;
        a[offa + 4] = x0r - x2r;
        a[offa + 5] = x0i - x2i;
        a[offa + 6] = x1r - x3i;
        a[offa + 7] = x1i + x3r;
    }

    public static void cftb040(DoubleLargeArray a, long offa)
    {
        double x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;

        x0r = a.getDouble(offa) + a.getDouble(offa + 4);
        x0i = a.getDouble(offa + 1) + a.getDouble(offa + 5);
        x1r = a.getDouble(offa) - a.getDouble(offa + 4);
        x1i = a.getDouble(offa + 1) - a.getDouble(offa + 5);
        x2r = a.getDouble(offa + 2) + a.getDouble(offa + 6);
        x2i = a.getDouble(offa + 3) + a.getDouble(offa + 7);
        x3r = a.getDouble(offa + 2) - a.getDouble(offa + 6);
        x3i = a.getDouble(offa + 3) - a.getDouble(offa + 7);
        a.setDouble(offa, x0r + x2r);
        a.setDouble(offa + 1, x0i + x2i);
        a.setDouble(offa + 2, x1r + x3i);
        a.setDouble(offa + 3, x1i - x3r);
        a.setDouble(offa + 4, x0r - x2r);
        a.setDouble(offa + 5, x0i - x2i);
        a.setDouble(offa + 6, x1r - x3i);
        a.setDouble(offa + 7, x1i + x3r);
    }

    public static void cftx020(double[] a, int offa)
    {
        double x0r, x0i;
        x0r = a[offa] - a[offa + 2];
        x0i = -a[offa + 1] + a[offa + 3];
        a[offa] += a[offa + 2];
        a[offa + 1] += a[offa + 3];
        a[offa + 2] = x0r;
        a[offa + 3] = x0i;
    }

    public static void cftx020(DoubleLargeArray a, long offa)
    {
        double x0r, x0i;
        x0r = a.getDouble(offa) - a.getDouble(offa + 2);
        x0i = -a.getDouble(offa + 1) + a.getDouble(offa + 3);
        a.setDouble(offa, a.getDouble(offa) + a.getDouble(offa + 2));
        a.setDouble(offa + 1, a.getDouble(offa + 1) + a.getDouble(offa + 3));
        a.setDouble(offa + 2, x0r);
        a.setDouble(offa + 3, x0i);
    }

    public static void cftxb020(double[] a, int offa)
    {
        double x0r, x0i;

        x0r = a[offa] - a[offa + 2];
        x0i = a[offa + 1] - a[offa + 3];
        a[offa] += a[offa + 2];
        a[offa + 1] += a[offa + 3];
        a[offa + 2] = x0r;
        a[offa + 3] = x0i;
    }

    public static void cftxb020(DoubleLargeArray a, long offa)
    {
        double x0r, x0i;

        x0r = a.getDouble(offa) - a.getDouble(offa + 2);
        x0i = a.getDouble(offa + 1) - a.getDouble(offa + 3);
        a.setDouble(offa, a.getDouble(offa) + a.getDouble(offa + 2));
        a.setDouble(offa + 1, a.getDouble(offa + 1) + a.getDouble(offa + 3));
        a.setDouble(offa + 2, x0r);
        a.setDouble(offa + 3, x0i);
    }

    public static void cftxc020(double[] a, int offa)
    {
        double x0r, x0i;
        x0r = a[offa] - a[offa + 2];
        x0i = a[offa + 1] + a[offa + 3];
        a[offa] += a[offa + 2];
        a[offa + 1] -= a[offa + 3];
        a[offa + 2] = x0r;
        a[offa + 3] = x0i;
    }

    public static void cftxc020(DoubleLargeArray a, long offa)
    {
        double x0r, x0i;
        x0r = a.getDouble(offa) - a.getDouble(offa + 2);
        x0i = a.getDouble(offa + 1) + a.getDouble(offa + 3);
        a.setDouble(offa, a.getDouble(offa) + a.getDouble(offa + 2));
        a.setDouble(offa + 1, a.getDouble(offa + 1) - a.getDouble(offa + 3));
        a.setDouble(offa + 2, x0r);
        a.setDouble(offa + 3, x0i);
    }

    public static void rftfsub(int n, double[] a, int offa, int nc, double[] c, int startc)
    {
        int k, kk, ks, m;
        double wkr, wki, xr, xi, yr, yi;
        int idx1, idx2;

        m = n >> 1;
        ks = 2 * nc / m;
        kk = 0;
        for (int j = 2; j < m; j += 2) {
            k = n - j;
            kk += ks;
            wkr = 0.5 - c[startc + nc - kk];
            wki = c[startc + kk];
            idx1 = offa + j;
            idx2 = offa + k;
            xr = a[idx1] - a[idx2];
            xi = a[idx1 + 1] + a[idx2 + 1];
            yr = wkr * xr - wki * xi;
            yi = wkr * xi + wki * xr;
            a[idx1] -= yr;
            a[idx1 + 1] = yi - a[idx1 + 1];
            a[idx2] += yr;
            a[idx2 + 1] = yi - a[idx2 + 1];
        }
        a[offa + m + 1] = -a[offa + m + 1];
    }

    public static void rftfsub(long n, DoubleLargeArray a, long offa, long nc, DoubleLargeArray c, long startc)
    {
        long k, kk, ks, m;
        double wkr, wki, xr, xi, yr, yi;
        long idx1, idx2;

        m = n >> 1l;
        ks = 2 * nc / m;
        kk = 0;
        for (long j = 2; j < m; j += 2) {
            k = n - j;
            kk += ks;
            wkr = 0.5 - c.getDouble(startc + nc - kk);
            wki = c.getDouble(startc + kk);
            idx1 = offa + j;
            idx2 = offa + k;
            xr = a.getDouble(idx1) - a.getDouble(idx2);
            xi = a.getDouble(idx1 + 1) + a.getDouble(idx2 + 1);
            yr = wkr * xr - wki * xi;
            yi = wkr * xi + wki * xr;
            a.setDouble(idx1, a.getDouble(idx1) - yr);
            a.setDouble(idx1 + 1, yi - a.getDouble(idx1 + 1));
            a.setDouble(idx2, a.getDouble(idx2) + yr);
            a.setDouble(idx2 + 1, yi - a.getDouble(idx2 + 1));
        }
        a.setDouble(offa + m + 1, -a.getDouble(offa + m + 1));
    }

    public static void rftbsub(int n, double[] a, int offa, int nc, double[] c, int startc)
    {
        int k, kk, ks, m;
        double wkr, wki, xr, xi, yr, yi;
        int idx1, idx2;

        m = n >> 1;
        ks = 2 * nc / m;
        kk = 0;
        for (int j = 2; j < m; j += 2) {
            k = n - j;
            kk += ks;
            wkr = 0.5 - c[startc + nc - kk];
            wki = c[startc + kk];
            idx1 = offa + j;
            idx2 = offa + k;
            xr = a[idx1] - a[idx2];
            xi = a[idx1 + 1] + a[idx2 + 1];
            yr = wkr * xr - wki * xi;
            yi = wkr * xi + wki * xr;
            a[idx1] -= yr;
            a[idx1 + 1] -= yi;
            a[idx2] += yr;
            a[idx2 + 1] -= yi;
        }
    }

    public static void rftbsub(long n, DoubleLargeArray a, long offa, long nc, DoubleLargeArray c, long startc)
    {
        long k, kk, ks, m;
        double wkr, wki, xr, xi, yr, yi;
        long idx1, idx2;

        m = n >> 1l;
        ks = 2 * nc / m;
        kk = 0;
        for (long j = 2; j < m; j += 2) {
            k = n - j;
            kk += ks;
            wkr = 0.5 - c.getDouble(startc + nc - kk);
            wki = c.getDouble(startc + kk);
            idx1 = offa + j;
            idx2 = offa + k;
            xr = a.getDouble(idx1) - a.getDouble(idx2);
            xi = a.getDouble(idx1 + 1) + a.getDouble(idx2 + 1);
            yr = wkr * xr - wki * xi;
            yi = wkr * xi + wki * xr;
            a.setDouble(idx1, a.getDouble(idx1) - yr);
            a.setDouble(idx1 + 1, a.getDouble(idx1 + 1) - yi);
            a.setDouble(idx2, a.getDouble(idx2) + yr);
            a.setDouble(idx2 + 1, a.getDouble(idx2 + 1) - yi);
        }
    }

    public static void dctsub(int n, double[] a, int offa, int nc, double[] c, int startc)
    {
        int k, kk, ks, m;
        double wkr, wki, xr;
        int idx0, idx1, idx2;

        m = n >> 1;
        ks = nc / n;
        kk = 0;
        for (int j = 1; j < m; j++) {
            k = n - j;
            kk += ks;
            idx0 = startc + kk;
            idx1 = offa + j;
            idx2 = offa + k;
            wkr = c[idx0] - c[startc + nc - kk];
            wki = c[idx0] + c[startc + nc - kk];
            xr = wki * a[idx1] - wkr * a[idx2];
            a[idx1] = wkr * a[idx1] + wki * a[idx2];
            a[idx2] = xr;
        }
        a[offa + m] *= c[startc];
    }

    public static void dctsub(long n, DoubleLargeArray a, long offa, long nc, DoubleLargeArray c, long startc)
    {
        long k, kk, ks, m;
        double wkr, wki, xr;
        long idx0, idx1, idx2;

        m = n >> 1l;
        ks = nc / n;
        kk = 0;
        for (long j = 1; j < m; j++) {
            k = n - j;
            kk += ks;
            idx0 = startc + kk;
            idx1 = offa + j;
            idx2 = offa + k;
            wkr = c.getDouble(idx0) - c.getDouble(startc + nc - kk);
            wki = c.getDouble(idx0) + c.getDouble(startc + nc - kk);
            xr = wki * a.getDouble(idx1) - wkr * a.getDouble(idx2);
            a.setDouble(idx1, wkr * a.getDouble(idx1) + wki * a.getDouble(idx2));
            a.setDouble(idx2, xr);
        }
        a.setDouble(offa + m, a.getDouble(offa + m) * c.getDouble(startc));
    }

    public static void cftfsub(int n, float[] a, int offa, int[] ip, int nw, float[] w)
    {
        if (n > 8) {
            if (n > 32) {
                cftf1st(n, a, offa, w, nw - (n >> 2));
                if ((ConcurrencyUtils.getNumberOfThreads() > 1) && (n >= CommonUtils.getThreadsBeginN_1D_FFT_2Threads())) {
                    cftrec4_th(n, a, offa, nw, w);
                } else if (n > 512) {
                    cftrec4(n, a, offa, nw, w);
                } else if (n > 128) {
                    cftleaf(n, 1, a, offa, nw, w);
                } else {
                    cftfx41(n, a, offa, nw, w);
                }
                bitrv2(n, ip, a, offa);
            } else if (n == 32) {
                cftf161(a, offa, w, nw - 8);
                bitrv216(a, offa);
            } else {
                cftf081(a, offa, w, 0);
                bitrv208(a, offa);
            }
        } else if (n == 8) {
            cftf040(a, offa);
        } else if (n == 4) {
            cftxb020(a, offa);
        }
    }

    public static void cftfsub(long n, FloatLargeArray a, long offa, LongLargeArray ip, long nw, FloatLargeArray w)
    {
        if (n > 8) {
            if (n > 32) {
                cftf1st(n, a, offa, w, nw - (n >> 2l));
                if ((ConcurrencyUtils.getNumberOfThreads() > 1) && (n >= CommonUtils.getThreadsBeginN_1D_FFT_2Threads())) {
                    cftrec4_th(n, a, offa, nw, w);
                } else if (n > 512) {
                    cftrec4(n, a, offa, nw, w);
                } else if (n > 128) {
                    cftleaf(n, 1, a, offa, nw, w);
                } else {
                    cftfx41(n, a, offa, nw, w);
                }
                bitrv2l(n, ip, a, offa);
            } else if (n == 32) {
                cftf161(a, offa, w, nw - 8);
                bitrv216(a, offa);
            } else {
                cftf081(a, offa, w, 0);
                bitrv208(a, offa);
            }
        } else if (n == 8) {
            cftf040(a, offa);
        } else if (n == 4) {
            cftxb020(a, offa);
        }
    }

    public static void cftbsub(int n, float[] a, int offa, int[] ip, int nw, float[] w)
    {
        if (n > 8) {
            if (n > 32) {
                cftb1st(n, a, offa, w, nw - (n >> 2));
                if ((ConcurrencyUtils.getNumberOfThreads() > 1) && (n >= CommonUtils.getThreadsBeginN_1D_FFT_2Threads())) {
                    cftrec4_th(n, a, offa, nw, w);
                } else if (n > 512) {
                    cftrec4(n, a, offa, nw, w);
                } else if (n > 128) {
                    cftleaf(n, 1, a, offa, nw, w);
                } else {
                    cftfx41(n, a, offa, nw, w);
                }
                bitrv2conj(n, ip, a, offa);
            } else if (n == 32) {
                cftf161(a, offa, w, nw - 8);
                bitrv216neg(a, offa);
            } else {
                cftf081(a, offa, w, 0);
                bitrv208neg(a, offa);
            }
        } else if (n == 8) {
            cftb040(a, offa);
        } else if (n == 4) {
            cftxb020(a, offa);
        }
    }

    public static void cftbsub(long n, FloatLargeArray a, long offa, LongLargeArray ip, long nw, FloatLargeArray w)
    {
        if (n > 8) {
            if (n > 32) {
                cftb1st(n, a, offa, w, nw - (n >> 2l));
                if ((ConcurrencyUtils.getNumberOfThreads() > 1) && (n >= CommonUtils.getThreadsBeginN_1D_FFT_2Threads())) {
                    cftrec4_th(n, a, offa, nw, w);
                } else if (n > 512) {
                    cftrec4(n, a, offa, nw, w);
                } else if (n > 128) {
                    cftleaf(n, 1, a, offa, nw, w);
                } else {
                    cftfx41(n, a, offa, nw, w);
                }
                bitrv2conj(n, ip, a, offa);
            } else if (n == 32) {
                cftf161(a, offa, w, nw - 8);
                bitrv216neg(a, offa);
            } else {
                cftf081(a, offa, w, 0);
                bitrv208neg(a, offa);
            }
        } else if (n == 8) {
            cftb040(a, offa);
        } else if (n == 4) {
            cftxb020(a, offa);
        }
    }

    public static void bitrv2(int n, int[] ip, float[] a, int offa)
    {
        int j1, k1, l, m, nh, nm;
        float xr, xi, yr, yi;
        int idx0, idx1, idx2;

        m = 1;
        for (l = n >> 2; l > 8; l >>= 2) {
            m <<= 1;
        }
        nh = n >> 1;
        nm = 4 * m;
        if (l == 8) {
            for (int k = 0; k < m; k++) {
                idx0 = 4 * k;
                for (int j = 0; j < k; j++) {
                    j1 = 4 * j + 2 * ip[m + k];
                    k1 = idx0 + 2 * ip[m + j];
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nh;
                    k1 += 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += 2;
                    k1 += nh;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nh;
                    k1 -= 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                }
                k1 = idx0 + 2 * ip[m + k];
                j1 = k1 + 2;
                k1 += nh;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a[idx1];
                xi = a[idx1 + 1];
                yr = a[idx2];
                yi = a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                j1 += nm;
                k1 += 2 * nm;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a[idx1];
                xi = a[idx1 + 1];
                yr = a[idx2];
                yi = a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                j1 += nm;
                k1 -= nm;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a[idx1];
                xi = a[idx1 + 1];
                yr = a[idx2];
                yi = a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                j1 -= 2;
                k1 -= nh;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a[idx1];
                xi = a[idx1 + 1];
                yr = a[idx2];
                yi = a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                j1 += nh + 2;
                k1 += nh + 2;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a[idx1];
                xi = a[idx1 + 1];
                yr = a[idx2];
                yi = a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                j1 -= nh - nm;
                k1 += 2 * nm - 2;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a[idx1];
                xi = a[idx1 + 1];
                yr = a[idx2];
                yi = a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
            }
        } else {
            for (int k = 0; k < m; k++) {
                idx0 = 4 * k;
                for (int j = 0; j < k; j++) {
                    j1 = 4 * j + ip[m + k];
                    k1 = idx0 + ip[m + j];
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nh;
                    k1 += 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += 2;
                    k1 += nh;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nh;
                    k1 -= 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = a[idx1 + 1];
                    yr = a[idx2];
                    yi = a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                }
                k1 = idx0 + ip[m + k];
                j1 = k1 + 2;
                k1 += nh;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a[idx1];
                xi = a[idx1 + 1];
                yr = a[idx2];
                yi = a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                j1 += nm;
                k1 += nm;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a[idx1];
                xi = a[idx1 + 1];
                yr = a[idx2];
                yi = a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
            }
        }
    }

    public static void bitrv2l(long n, LongLargeArray ip, FloatLargeArray a, long offa)
    {
        long j1, k1, l, m, nh, nm;
        float xr, xi, yr, yi;
        long idx0, idx1, idx2;

        m = 1;
        for (l = n >> 2l; l > 8; l >>= 2l) {
            m <<= 1l;
        }
        nh = n >> 1l;
        nm = 4 * m;
        if (l == 8) {
            for (long k = 0; k < m; k++) {
                idx0 = 4 * k;
                for (long j = 0; j < k; j++) {
                    j1 = 4 * j + 2 * ip.getLong(m + k);
                    k1 = idx0 + 2 * ip.getLong(m + j);
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 += nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 += nh;
                    k1 += 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 -= nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 += 2;
                    k1 += nh;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 += nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 -= nh;
                    k1 -= 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 -= nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                }
                k1 = idx0 + 2 * ip.getLong(m + k);
                j1 = k1 + 2;
                k1 += nh;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a.getFloat(idx1);
                xi = a.getFloat(idx1 + 1);
                yr = a.getFloat(idx2);
                yi = a.getFloat(idx2 + 1);
                a.setFloat(idx1, yr);
                a.setFloat(idx1 + 1, yi);
                a.setFloat(idx2, xr);
                a.setFloat(idx2 + 1, xi);
                j1 += nm;
                k1 += 2 * nm;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a.getFloat(idx1);
                xi = a.getFloat(idx1 + 1);
                yr = a.getFloat(idx2);
                yi = a.getFloat(idx2 + 1);
                a.setFloat(idx1, yr);
                a.setFloat(idx1 + 1, yi);
                a.setFloat(idx2, xr);
                a.setFloat(idx2 + 1, xi);
                j1 += nm;
                k1 -= nm;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a.getFloat(idx1);
                xi = a.getFloat(idx1 + 1);
                yr = a.getFloat(idx2);
                yi = a.getFloat(idx2 + 1);
                a.setFloat(idx1, yr);
                a.setFloat(idx1 + 1, yi);
                a.setFloat(idx2, xr);
                a.setFloat(idx2 + 1, xi);
                j1 -= 2;
                k1 -= nh;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a.getFloat(idx1);
                xi = a.getFloat(idx1 + 1);
                yr = a.getFloat(idx2);
                yi = a.getFloat(idx2 + 1);
                a.setFloat(idx1, yr);
                a.setFloat(idx1 + 1, yi);
                a.setFloat(idx2, xr);
                a.setFloat(idx2 + 1, xi);
                j1 += nh + 2;
                k1 += nh + 2;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a.getFloat(idx1);
                xi = a.getFloat(idx1 + 1);
                yr = a.getFloat(idx2);
                yi = a.getFloat(idx2 + 1);
                a.setFloat(idx1, yr);
                a.setFloat(idx1 + 1, yi);
                a.setFloat(idx2, xr);
                a.setFloat(idx2 + 1, xi);
                j1 -= nh - nm;
                k1 += 2 * nm - 2;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a.getFloat(idx1);
                xi = a.getFloat(idx1 + 1);
                yr = a.getFloat(idx2);
                yi = a.getFloat(idx2 + 1);
                a.setFloat(idx1, yr);
                a.setFloat(idx1 + 1, yi);
                a.setFloat(idx2, xr);
                a.setFloat(idx2 + 1, xi);
            }
        } else {
            for (long k = 0; k < m; k++) {
                idx0 = 4 * k;
                for (long j = 0; j < k; j++) {
                    j1 = 4 * j + ip.getLong(m + k);
                    k1 = idx0 + ip.getLong(m + j);
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 += nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 += nh;
                    k1 += 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 -= nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 += 2;
                    k1 += nh;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 += nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 -= nh;
                    k1 -= 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 -= nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                }
                k1 = idx0 + ip.getLong(m + k);
                j1 = k1 + 2;
                k1 += nh;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a.getFloat(idx1);
                xi = a.getFloat(idx1 + 1);
                yr = a.getFloat(idx2);
                yi = a.getFloat(idx2 + 1);
                a.setFloat(idx1, yr);
                a.setFloat(idx1 + 1, yi);
                a.setFloat(idx2, xr);
                a.setFloat(idx2 + 1, xi);
                j1 += nm;
                k1 += nm;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a.getFloat(idx1);
                xi = a.getFloat(idx1 + 1);
                yr = a.getFloat(idx2);
                yi = a.getFloat(idx2 + 1);
                a.setFloat(idx1, yr);
                a.setFloat(idx1 + 1, yi);
                a.setFloat(idx2, xr);
                a.setFloat(idx2 + 1, xi);
            }
        }
    }

    public static void bitrv2conj(int n, int[] ip, float[] a, int offa)
    {
        int j1, k1, l, m, nh, nm;
        float xr, xi, yr, yi;
        int idx0, idx1, idx2;

        m = 1;
        for (l = n >> 2; l > 8; l >>= 2) {
            m <<= 1;
        }
        nh = n >> 1;
        nm = 4 * m;
        if (l == 8) {
            for (int k = 0; k < m; k++) {
                idx0 = 4 * k;
                for (int j = 0; j < k; j++) {
                    j1 = 4 * j + 2 * ip[m + k];
                    k1 = idx0 + 2 * ip[m + j];
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nh;
                    k1 += 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += 2;
                    k1 += nh;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nh;
                    k1 -= 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                }
                k1 = idx0 + 2 * ip[m + k];
                j1 = k1 + 2;
                k1 += nh;
                idx1 = offa + j1;
                idx2 = offa + k1;
                a[idx1 - 1] = -a[idx1 - 1];
                xr = a[idx1];
                xi = -a[idx1 + 1];
                yr = a[idx2];
                yi = -a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                a[idx2 + 3] = -a[idx2 + 3];
                j1 += nm;
                k1 += 2 * nm;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a[idx1];
                xi = -a[idx1 + 1];
                yr = a[idx2];
                yi = -a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                j1 += nm;
                k1 -= nm;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a[idx1];
                xi = -a[idx1 + 1];
                yr = a[idx2];
                yi = -a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                j1 -= 2;
                k1 -= nh;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a[idx1];
                xi = -a[idx1 + 1];
                yr = a[idx2];
                yi = -a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                j1 += nh + 2;
                k1 += nh + 2;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a[idx1];
                xi = -a[idx1 + 1];
                yr = a[idx2];
                yi = -a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                j1 -= nh - nm;
                k1 += 2 * nm - 2;
                idx1 = offa + j1;
                idx2 = offa + k1;
                a[idx1 - 1] = -a[idx1 - 1];
                xr = a[idx1];
                xi = -a[idx1 + 1];
                yr = a[idx2];
                yi = -a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                a[idx2 + 3] = -a[idx2 + 3];
            }
        } else {
            for (int k = 0; k < m; k++) {
                idx0 = 4 * k;
                for (int j = 0; j < k; j++) {
                    j1 = 4 * j + ip[m + k];
                    k1 = idx0 + ip[m + j];
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nh;
                    k1 += 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += 2;
                    k1 += nh;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 += nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nh;
                    k1 -= 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                    j1 -= nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a[idx1];
                    xi = -a[idx1 + 1];
                    yr = a[idx2];
                    yi = -a[idx2 + 1];
                    a[idx1] = yr;
                    a[idx1 + 1] = yi;
                    a[idx2] = xr;
                    a[idx2 + 1] = xi;
                }
                k1 = idx0 + ip[m + k];
                j1 = k1 + 2;
                k1 += nh;
                idx1 = offa + j1;
                idx2 = offa + k1;
                a[idx1 - 1] = -a[idx1 - 1];
                xr = a[idx1];
                xi = -a[idx1 + 1];
                yr = a[idx2];
                yi = -a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                a[idx2 + 3] = -a[idx2 + 3];
                j1 += nm;
                k1 += nm;
                idx1 = offa + j1;
                idx2 = offa + k1;
                a[idx1 - 1] = -a[idx1 - 1];
                xr = a[idx1];
                xi = -a[idx1 + 1];
                yr = a[idx2];
                yi = -a[idx2 + 1];
                a[idx1] = yr;
                a[idx1 + 1] = yi;
                a[idx2] = xr;
                a[idx2 + 1] = xi;
                a[idx2 + 3] = -a[idx2 + 3];
            }
        }
    }

    public static void bitrv2conj(long n, LongLargeArray ip, FloatLargeArray a, long offa)
    {
        long j1, k1, l, m, nh, nm;
        float xr, xi, yr, yi;
        long idx0, idx1, idx2;

        m = 1;
        for (l = n >> 2l; l > 8; l >>= 2l) {
            m <<= 1;
        }
        nh = n >> 1l;
        nm = 4 * m;
        if (l == 8) {
            for (long k = 0; k < m; k++) {
                idx0 = 4 * k;
                for (long j = 0; j < k; j++) {
                    j1 = 4 * j + 2 * ip.getLong(m + k);
                    k1 = idx0 + 2 * ip.getLong(m + j);
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = -a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = -a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = -a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = -a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 += nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = -a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = -a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = -a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = -a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 += nh;
                    k1 += 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = -a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = -a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = -a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = -a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 -= nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = -a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = -a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = -a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = -a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 += 2;
                    k1 += nh;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = -a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = -a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = -a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = -a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 += nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = -a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = -a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 += nm;
                    k1 += 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = -a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = -a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 -= nh;
                    k1 -= 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = -a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = -a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = -a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = -a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 -= nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = -a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = -a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 -= nm;
                    k1 -= 2 * nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = -a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = -a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                }
                k1 = idx0 + 2 * ip.getLong(m + k);
                j1 = k1 + 2;
                k1 += nh;
                idx1 = offa + j1;
                idx2 = offa + k1;
                a.setFloat(idx1 - 1, -a.getFloat(idx1 - 1));
                xr = a.getFloat(idx1);
                xi = -a.getFloat(idx1 + 1);
                yr = a.getFloat(idx2);
                yi = -a.getFloat(idx2 + 1);
                a.setFloat(idx1, yr);
                a.setFloat(idx1 + 1, yi);
                a.setFloat(idx2, xr);
                a.setFloat(idx2 + 1, xi);
                a.setFloat(idx2 + 3, -a.getFloat(idx2 + 3));
                j1 += nm;
                k1 += 2 * nm;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a.getFloat(idx1);
                xi = -a.getFloat(idx1 + 1);
                yr = a.getFloat(idx2);
                yi = -a.getFloat(idx2 + 1);
                a.setFloat(idx1, yr);
                a.setFloat(idx1 + 1, yi);
                a.setFloat(idx2, xr);
                a.setFloat(idx2 + 1, xi);
                j1 += nm;
                k1 -= nm;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a.getFloat(idx1);
                xi = -a.getFloat(idx1 + 1);
                yr = a.getFloat(idx2);
                yi = -a.getFloat(idx2 + 1);
                a.setFloat(idx1, yr);
                a.setFloat(idx1 + 1, yi);
                a.setFloat(idx2, xr);
                a.setFloat(idx2 + 1, xi);
                j1 -= 2;
                k1 -= nh;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a.getFloat(idx1);
                xi = -a.getFloat(idx1 + 1);
                yr = a.getFloat(idx2);
                yi = -a.getFloat(idx2 + 1);
                a.setFloat(idx1, yr);
                a.setFloat(idx1 + 1, yi);
                a.setFloat(idx2, xr);
                a.setFloat(idx2 + 1, xi);
                j1 += nh + 2;
                k1 += nh + 2;
                idx1 = offa + j1;
                idx2 = offa + k1;
                xr = a.getFloat(idx1);
                xi = -a.getFloat(idx1 + 1);
                yr = a.getFloat(idx2);
                yi = -a.getFloat(idx2 + 1);
                a.setFloat(idx1, yr);
                a.setFloat(idx1 + 1, yi);
                a.setFloat(idx2, xr);
                a.setFloat(idx2 + 1, xi);
                j1 -= nh - nm;
                k1 += 2 * nm - 2;
                idx1 = offa + j1;
                idx2 = offa + k1;
                a.setFloat(idx1 - 1, -a.getFloat(idx1 - 1));
                xr = a.getFloat(idx1);
                xi = -a.getFloat(idx1 + 1);
                yr = a.getFloat(idx2);
                yi = -a.getFloat(idx2 + 1);
                a.setFloat(idx1, yr);
                a.setFloat(idx1 + 1, yi);
                a.setFloat(idx2, xr);
                a.setFloat(idx2 + 1, xi);
                a.setFloat(idx2 + 3, -a.getFloat(idx2 + 3));
            }
        } else {
            for (int k = 0; k < m; k++) {
                idx0 = 4 * k;
                for (int j = 0; j < k; j++) {
                    j1 = 4 * j + ip.getLong(m + k);
                    k1 = idx0 + ip.getLong(m + j);
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = -a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = -a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 += nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = -a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = -a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 += nh;
                    k1 += 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = -a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = -a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 -= nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = -a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = -a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 += 2;
                    k1 += nh;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = -a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = -a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 += nm;
                    k1 += nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = -a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = -a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 -= nh;
                    k1 -= 2;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = -a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = -a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                    j1 -= nm;
                    k1 -= nm;
                    idx1 = offa + j1;
                    idx2 = offa + k1;
                    xr = a.getFloat(idx1);
                    xi = -a.getFloat(idx1 + 1);
                    yr = a.getFloat(idx2);
                    yi = -a.getFloat(idx2 + 1);
                    a.setFloat(idx1, yr);
                    a.setFloat(idx1 + 1, yi);
                    a.setFloat(idx2, xr);
                    a.setFloat(idx2 + 1, xi);
                }
                k1 = idx0 + ip.getLong(m + k);
                j1 = k1 + 2;
                k1 += nh;
                idx1 = offa + j1;
                idx2 = offa + k1;
                a.setFloat(idx1 - 1, -a.getFloat(idx1 - 1));
                xr = a.getFloat(idx1);
                xi = -a.getFloat(idx1 + 1);
                yr = a.getFloat(idx2);
                yi = -a.getFloat(idx2 + 1);
                a.setFloat(idx1, yr);
                a.setFloat(idx1 + 1, yi);
                a.setFloat(idx2, xr);
                a.setFloat(idx2 + 1, xi);
                a.setFloat(idx2 + 3, -a.getFloat(idx2 + 3));
                j1 += nm;
                k1 += nm;
                idx1 = offa + j1;
                idx2 = offa + k1;
                a.setFloat(idx1 - 1, -a.getFloat(idx1 - 1));
                xr = a.getFloat(idx1);
                xi = -a.getFloat(idx1 + 1);
                yr = a.getFloat(idx2);
                yi = -a.getFloat(idx2 + 1);
                a.setFloat(idx1, yr);
                a.setFloat(idx1 + 1, yi);
                a.setFloat(idx2, xr);
                a.setFloat(idx2 + 1, xi);
                a.setFloat(idx2 + 3, -a.getFloat(idx2 + 3));
            }
        }
    }

    public static void bitrv216(float[] a, int offa)
    {
        float x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, x5r, x5i, x7r, x7i, x8r, x8i, x10r, x10i, x11r, x11i, x12r, x12i, x13r, x13i, x14r, x14i;

        x1r = a[offa + 2];
        x1i = a[offa + 3];
        x2r = a[offa + 4];
        x2i = a[offa + 5];
        x3r = a[offa + 6];
        x3i = a[offa + 7];
        x4r = a[offa + 8];
        x4i = a[offa + 9];
        x5r = a[offa + 10];
        x5i = a[offa + 11];
        x7r = a[offa + 14];
        x7i = a[offa + 15];
        x8r = a[offa + 16];
        x8i = a[offa + 17];
        x10r = a[offa + 20];
        x10i = a[offa + 21];
        x11r = a[offa + 22];
        x11i = a[offa + 23];
        x12r = a[offa + 24];
        x12i = a[offa + 25];
        x13r = a[offa + 26];
        x13i = a[offa + 27];
        x14r = a[offa + 28];
        x14i = a[offa + 29];
        a[offa + 2] = x8r;
        a[offa + 3] = x8i;
        a[offa + 4] = x4r;
        a[offa + 5] = x4i;
        a[offa + 6] = x12r;
        a[offa + 7] = x12i;
        a[offa + 8] = x2r;
        a[offa + 9] = x2i;
        a[offa + 10] = x10r;
        a[offa + 11] = x10i;
        a[offa + 14] = x14r;
        a[offa + 15] = x14i;
        a[offa + 16] = x1r;
        a[offa + 17] = x1i;
        a[offa + 20] = x5r;
        a[offa + 21] = x5i;
        a[offa + 22] = x13r;
        a[offa + 23] = x13i;
        a[offa + 24] = x3r;
        a[offa + 25] = x3i;
        a[offa + 26] = x11r;
        a[offa + 27] = x11i;
        a[offa + 28] = x7r;
        a[offa + 29] = x7i;
    }

    public static void bitrv216(FloatLargeArray a, long offa)
    {
        float x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, x5r, x5i, x7r, x7i, x8r, x8i, x10r, x10i, x11r, x11i, x12r, x12i, x13r, x13i, x14r, x14i;

        x1r = a.getFloat(offa + 2);
        x1i = a.getFloat(offa + 3);
        x2r = a.getFloat(offa + 4);
        x2i = a.getFloat(offa + 5);
        x3r = a.getFloat(offa + 6);
        x3i = a.getFloat(offa + 7);
        x4r = a.getFloat(offa + 8);
        x4i = a.getFloat(offa + 9);
        x5r = a.getFloat(offa + 10);
        x5i = a.getFloat(offa + 11);
        x7r = a.getFloat(offa + 14);
        x7i = a.getFloat(offa + 15);
        x8r = a.getFloat(offa + 16);
        x8i = a.getFloat(offa + 17);
        x10r = a.getFloat(offa + 20);
        x10i = a.getFloat(offa + 21);
        x11r = a.getFloat(offa + 22);
        x11i = a.getFloat(offa + 23);
        x12r = a.getFloat(offa + 24);
        x12i = a.getFloat(offa + 25);
        x13r = a.getFloat(offa + 26);
        x13i = a.getFloat(offa + 27);
        x14r = a.getFloat(offa + 28);
        x14i = a.getFloat(offa + 29);
        a.setFloat(offa + 2, x8r);
        a.setFloat(offa + 3, x8i);
        a.setFloat(offa + 4, x4r);
        a.setFloat(offa + 5, x4i);
        a.setFloat(offa + 6, x12r);
        a.setFloat(offa + 7, x12i);
        a.setFloat(offa + 8, x2r);
        a.setFloat(offa + 9, x2i);
        a.setFloat(offa + 10, x10r);
        a.setFloat(offa + 11, x10i);
        a.setFloat(offa + 14, x14r);
        a.setFloat(offa + 15, x14i);
        a.setFloat(offa + 16, x1r);
        a.setFloat(offa + 17, x1i);
        a.setFloat(offa + 20, x5r);
        a.setFloat(offa + 21, x5i);
        a.setFloat(offa + 22, x13r);
        a.setFloat(offa + 23, x13i);
        a.setFloat(offa + 24, x3r);
        a.setFloat(offa + 25, x3i);
        a.setFloat(offa + 26, x11r);
        a.setFloat(offa + 27, x11i);
        a.setFloat(offa + 28, x7r);
        a.setFloat(offa + 29, x7i);
    }

    public static void bitrv216neg(float[] a, int offa)
    {
        float x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i, x8r, x8i, x9r, x9i, x10r, x10i, x11r, x11i, x12r, x12i, x13r, x13i, x14r, x14i, x15r, x15i;

        x1r = a[offa + 2];
        x1i = a[offa + 3];
        x2r = a[offa + 4];
        x2i = a[offa + 5];
        x3r = a[offa + 6];
        x3i = a[offa + 7];
        x4r = a[offa + 8];
        x4i = a[offa + 9];
        x5r = a[offa + 10];
        x5i = a[offa + 11];
        x6r = a[offa + 12];
        x6i = a[offa + 13];
        x7r = a[offa + 14];
        x7i = a[offa + 15];
        x8r = a[offa + 16];
        x8i = a[offa + 17];
        x9r = a[offa + 18];
        x9i = a[offa + 19];
        x10r = a[offa + 20];
        x10i = a[offa + 21];
        x11r = a[offa + 22];
        x11i = a[offa + 23];
        x12r = a[offa + 24];
        x12i = a[offa + 25];
        x13r = a[offa + 26];
        x13i = a[offa + 27];
        x14r = a[offa + 28];
        x14i = a[offa + 29];
        x15r = a[offa + 30];
        x15i = a[offa + 31];
        a[offa + 2] = x15r;
        a[offa + 3] = x15i;
        a[offa + 4] = x7r;
        a[offa + 5] = x7i;
        a[offa + 6] = x11r;
        a[offa + 7] = x11i;
        a[offa + 8] = x3r;
        a[offa + 9] = x3i;
        a[offa + 10] = x13r;
        a[offa + 11] = x13i;
        a[offa + 12] = x5r;
        a[offa + 13] = x5i;
        a[offa + 14] = x9r;
        a[offa + 15] = x9i;
        a[offa + 16] = x1r;
        a[offa + 17] = x1i;
        a[offa + 18] = x14r;
        a[offa + 19] = x14i;
        a[offa + 20] = x6r;
        a[offa + 21] = x6i;
        a[offa + 22] = x10r;
        a[offa + 23] = x10i;
        a[offa + 24] = x2r;
        a[offa + 25] = x2i;
        a[offa + 26] = x12r;
        a[offa + 27] = x12i;
        a[offa + 28] = x4r;
        a[offa + 29] = x4i;
        a[offa + 30] = x8r;
        a[offa + 31] = x8i;
    }

    public static void bitrv216neg(FloatLargeArray a, long offa)
    {
        float x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i, x8r, x8i, x9r, x9i, x10r, x10i, x11r, x11i, x12r, x12i, x13r, x13i, x14r, x14i, x15r, x15i;

        x1r = a.getFloat(offa + 2);
        x1i = a.getFloat(offa + 3);
        x2r = a.getFloat(offa + 4);
        x2i = a.getFloat(offa + 5);
        x3r = a.getFloat(offa + 6);
        x3i = a.getFloat(offa + 7);
        x4r = a.getFloat(offa + 8);
        x4i = a.getFloat(offa + 9);
        x5r = a.getFloat(offa + 10);
        x5i = a.getFloat(offa + 11);
        x6r = a.getFloat(offa + 12);
        x6i = a.getFloat(offa + 13);
        x7r = a.getFloat(offa + 14);
        x7i = a.getFloat(offa + 15);
        x8r = a.getFloat(offa + 16);
        x8i = a.getFloat(offa + 17);
        x9r = a.getFloat(offa + 18);
        x9i = a.getFloat(offa + 19);
        x10r = a.getFloat(offa + 20);
        x10i = a.getFloat(offa + 21);
        x11r = a.getFloat(offa + 22);
        x11i = a.getFloat(offa + 23);
        x12r = a.getFloat(offa + 24);
        x12i = a.getFloat(offa + 25);
        x13r = a.getFloat(offa + 26);
        x13i = a.getFloat(offa + 27);
        x14r = a.getFloat(offa + 28);
        x14i = a.getFloat(offa + 29);
        x15r = a.getFloat(offa + 30);
        x15i = a.getFloat(offa + 31);
        a.setFloat(offa + 2, x15r);
        a.setFloat(offa + 3, x15i);
        a.setFloat(offa + 4, x7r);
        a.setFloat(offa + 5, x7i);
        a.setFloat(offa + 6, x11r);
        a.setFloat(offa + 7, x11i);
        a.setFloat(offa + 8, x3r);
        a.setFloat(offa + 9, x3i);
        a.setFloat(offa + 10, x13r);
        a.setFloat(offa + 11, x13i);
        a.setFloat(offa + 12, x5r);
        a.setFloat(offa + 13, x5i);
        a.setFloat(offa + 14, x9r);
        a.setFloat(offa + 15, x9i);
        a.setFloat(offa + 16, x1r);
        a.setFloat(offa + 17, x1i);
        a.setFloat(offa + 18, x14r);
        a.setFloat(offa + 19, x14i);
        a.setFloat(offa + 20, x6r);
        a.setFloat(offa + 21, x6i);
        a.setFloat(offa + 22, x10r);
        a.setFloat(offa + 23, x10i);
        a.setFloat(offa + 24, x2r);
        a.setFloat(offa + 25, x2i);
        a.setFloat(offa + 26, x12r);
        a.setFloat(offa + 27, x12i);
        a.setFloat(offa + 28, x4r);
        a.setFloat(offa + 29, x4i);
        a.setFloat(offa + 30, x8r);
        a.setFloat(offa + 31, x8i);
    }

    public static void bitrv208(float[] a, int offa)
    {
        float x1r, x1i, x3r, x3i, x4r, x4i, x6r, x6i;

        x1r = a[offa + 2];
        x1i = a[offa + 3];
        x3r = a[offa + 6];
        x3i = a[offa + 7];
        x4r = a[offa + 8];
        x4i = a[offa + 9];
        x6r = a[offa + 12];
        x6i = a[offa + 13];
        a[offa + 2] = x4r;
        a[offa + 3] = x4i;
        a[offa + 6] = x6r;
        a[offa + 7] = x6i;
        a[offa + 8] = x1r;
        a[offa + 9] = x1i;
        a[offa + 12] = x3r;
        a[offa + 13] = x3i;
    }

    public static void bitrv208(FloatLargeArray a, long offa)
    {
        float x1r, x1i, x3r, x3i, x4r, x4i, x6r, x6i;

        x1r = a.getFloat(offa + 2);
        x1i = a.getFloat(offa + 3);
        x3r = a.getFloat(offa + 6);
        x3i = a.getFloat(offa + 7);
        x4r = a.getFloat(offa + 8);
        x4i = a.getFloat(offa + 9);
        x6r = a.getFloat(offa + 12);
        x6i = a.getFloat(offa + 13);
        a.setFloat(offa + 2, x4r);
        a.setFloat(offa + 3, x4i);
        a.setFloat(offa + 6, x6r);
        a.setFloat(offa + 7, x6i);
        a.setFloat(offa + 8, x1r);
        a.setFloat(offa + 9, x1i);
        a.setFloat(offa + 12, x3r);
        a.setFloat(offa + 13, x3i);
    }

    public static void bitrv208neg(float[] a, int offa)
    {
        float x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i;

        x1r = a[offa + 2];
        x1i = a[offa + 3];
        x2r = a[offa + 4];
        x2i = a[offa + 5];
        x3r = a[offa + 6];
        x3i = a[offa + 7];
        x4r = a[offa + 8];
        x4i = a[offa + 9];
        x5r = a[offa + 10];
        x5i = a[offa + 11];
        x6r = a[offa + 12];
        x6i = a[offa + 13];
        x7r = a[offa + 14];
        x7i = a[offa + 15];
        a[offa + 2] = x7r;
        a[offa + 3] = x7i;
        a[offa + 4] = x3r;
        a[offa + 5] = x3i;
        a[offa + 6] = x5r;
        a[offa + 7] = x5i;
        a[offa + 8] = x1r;
        a[offa + 9] = x1i;
        a[offa + 10] = x6r;
        a[offa + 11] = x6i;
        a[offa + 12] = x2r;
        a[offa + 13] = x2i;
        a[offa + 14] = x4r;
        a[offa + 15] = x4i;
    }

    public static void bitrv208neg(FloatLargeArray a, long offa)
    {
        float x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i;

        x1r = a.getFloat(offa + 2);
        x1i = a.getFloat(offa + 3);
        x2r = a.getFloat(offa + 4);
        x2i = a.getFloat(offa + 5);
        x3r = a.getFloat(offa + 6);
        x3i = a.getFloat(offa + 7);
        x4r = a.getFloat(offa + 8);
        x4i = a.getFloat(offa + 9);
        x5r = a.getFloat(offa + 10);
        x5i = a.getFloat(offa + 11);
        x6r = a.getFloat(offa + 12);
        x6i = a.getFloat(offa + 13);
        x7r = a.getFloat(offa + 14);
        x7i = a.getFloat(offa + 15);
        a.setFloat(offa + 2, x7r);
        a.setFloat(offa + 3, x7i);
        a.setFloat(offa + 4, x3r);
        a.setFloat(offa + 5, x3i);
        a.setFloat(offa + 6, x5r);
        a.setFloat(offa + 7, x5i);
        a.setFloat(offa + 8, x1r);
        a.setFloat(offa + 9, x1i);
        a.setFloat(offa + 10, x6r);
        a.setFloat(offa + 11, x6i);
        a.setFloat(offa + 12, x2r);
        a.setFloat(offa + 13, x2i);
        a.setFloat(offa + 14, x4r);
        a.setFloat(offa + 15, x4i);
    }

    public static void cftf1st(int n, float[] a, int offa, float[] w, int startw)
    {
        int j0, j1, j2, j3, k, m, mh;
        float wn4r, csc1, csc3, wk1r, wk1i, wk3r, wk3i, wd1r, wd1i, wd3r, wd3i;
        float x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        int idx0, idx1, idx2, idx3, idx4, idx5;
        mh = n >> 3;
        m = 2 * mh;
        j1 = m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a[offa] + a[idx2];
        x0i = a[offa + 1] + a[idx2 + 1];
        x1r = a[offa] - a[idx2];
        x1i = a[offa + 1] - a[idx2 + 1];
        x2r = a[idx1] + a[idx3];
        x2i = a[idx1 + 1] + a[idx3 + 1];
        x3r = a[idx1] - a[idx3];
        x3i = a[idx1 + 1] - a[idx3 + 1];
        a[offa] = x0r + x2r;
        a[offa + 1] = x0i + x2i;
        a[idx1] = x0r - x2r;
        a[idx1 + 1] = x0i - x2i;
        a[idx2] = x1r - x3i;
        a[idx2 + 1] = x1i + x3r;
        a[idx3] = x1r + x3i;
        a[idx3 + 1] = x1i - x3r;
        wn4r = w[startw + 1];
        csc1 = w[startw + 2];
        csc3 = w[startw + 3];
        wd1r = 1;
        wd1i = 0;
        wd3r = 1;
        wd3i = 0;
        k = 0;
        for (int j = 2; j < mh - 2; j += 4) {
            k += 4;
            idx4 = startw + k;
            wk1r = csc1 * (wd1r + w[idx4]);
            wk1i = csc1 * (wd1i + w[idx4 + 1]);
            wk3r = csc3 * (wd3r + w[idx4 + 2]);
            wk3i = csc3 * (wd3i + w[idx4 + 3]);
            wd1r = w[idx4];
            wd1i = w[idx4 + 1];
            wd3r = w[idx4 + 2];
            wd3i = w[idx4 + 3];
            j1 = j + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            idx5 = offa + j;
            x0r = a[idx5] + a[idx2];
            x0i = a[idx5 + 1] + a[idx2 + 1];
            x1r = a[idx5] - a[idx2];
            x1i = a[idx5 + 1] - a[idx2 + 1];
            y0r = a[idx5 + 2] + a[idx2 + 2];
            y0i = a[idx5 + 3] + a[idx2 + 3];
            y1r = a[idx5 + 2] - a[idx2 + 2];
            y1i = a[idx5 + 3] - a[idx2 + 3];
            x2r = a[idx1] + a[idx3];
            x2i = a[idx1 + 1] + a[idx3 + 1];
            x3r = a[idx1] - a[idx3];
            x3i = a[idx1 + 1] - a[idx3 + 1];
            y2r = a[idx1 + 2] + a[idx3 + 2];
            y2i = a[idx1 + 3] + a[idx3 + 3];
            y3r = a[idx1 + 2] - a[idx3 + 2];
            y3i = a[idx1 + 3] - a[idx3 + 3];
            a[idx5] = x0r + x2r;
            a[idx5 + 1] = x0i + x2i;
            a[idx5 + 2] = y0r + y2r;
            a[idx5 + 3] = y0i + y2i;
            a[idx1] = x0r - x2r;
            a[idx1 + 1] = x0i - x2i;
            a[idx1 + 2] = y0r - y2r;
            a[idx1 + 3] = y0i - y2i;
            x0r = x1r - x3i;
            x0i = x1i + x3r;
            a[idx2] = wk1r * x0r - wk1i * x0i;
            a[idx2 + 1] = wk1r * x0i + wk1i * x0r;
            x0r = y1r - y3i;
            x0i = y1i + y3r;
            a[idx2 + 2] = wd1r * x0r - wd1i * x0i;
            a[idx2 + 3] = wd1r * x0i + wd1i * x0r;
            x0r = x1r + x3i;
            x0i = x1i - x3r;
            a[idx3] = wk3r * x0r + wk3i * x0i;
            a[idx3 + 1] = wk3r * x0i - wk3i * x0r;
            x0r = y1r + y3i;
            x0i = y1i - y3r;
            a[idx3 + 2] = wd3r * x0r + wd3i * x0i;
            a[idx3 + 3] = wd3r * x0i - wd3i * x0r;
            j0 = m - j;
            j1 = j0 + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx0 = offa + j0;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            x0r = a[idx0] + a[idx2];
            x0i = a[idx0 + 1] + a[idx2 + 1];
            x1r = a[idx0] - a[idx2];
            x1i = a[idx0 + 1] - a[idx2 + 1];
            y0r = a[idx0 - 2] + a[idx2 - 2];
            y0i = a[idx0 - 1] + a[idx2 - 1];
            y1r = a[idx0 - 2] - a[idx2 - 2];
            y1i = a[idx0 - 1] - a[idx2 - 1];
            x2r = a[idx1] + a[idx3];
            x2i = a[idx1 + 1] + a[idx3 + 1];
            x3r = a[idx1] - a[idx3];
            x3i = a[idx1 + 1] - a[idx3 + 1];
            y2r = a[idx1 - 2] + a[idx3 - 2];
            y2i = a[idx1 - 1] + a[idx3 - 1];
            y3r = a[idx1 - 2] - a[idx3 - 2];
            y3i = a[idx1 - 1] - a[idx3 - 1];
            a[idx0] = x0r + x2r;
            a[idx0 + 1] = x0i + x2i;
            a[idx0 - 2] = y0r + y2r;
            a[idx0 - 1] = y0i + y2i;
            a[idx1] = x0r - x2r;
            a[idx1 + 1] = x0i - x2i;
            a[idx1 - 2] = y0r - y2r;
            a[idx1 - 1] = y0i - y2i;
            x0r = x1r - x3i;
            x0i = x1i + x3r;
            a[idx2] = wk1i * x0r - wk1r * x0i;
            a[idx2 + 1] = wk1i * x0i + wk1r * x0r;
            x0r = y1r - y3i;
            x0i = y1i + y3r;
            a[idx2 - 2] = wd1i * x0r - wd1r * x0i;
            a[idx2 - 1] = wd1i * x0i + wd1r * x0r;
            x0r = x1r + x3i;
            x0i = x1i - x3r;
            a[idx3] = wk3i * x0r + wk3r * x0i;
            a[idx3 + 1] = wk3i * x0i - wk3r * x0r;
            x0r = y1r + y3i;
            x0i = y1i - y3r;
            a[offa + j3 - 2] = wd3i * x0r + wd3r * x0i;
            a[offa + j3 - 1] = wd3i * x0i - wd3r * x0r;
        }
        wk1r = csc1 * (wd1r + wn4r);
        wk1i = csc1 * (wd1i + wn4r);
        wk3r = csc3 * (wd3r - wn4r);
        wk3i = csc3 * (wd3i - wn4r);
        j0 = mh;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx0 = offa + j0;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a[idx0 - 2] + a[idx2 - 2];
        x0i = a[idx0 - 1] + a[idx2 - 1];
        x1r = a[idx0 - 2] - a[idx2 - 2];
        x1i = a[idx0 - 1] - a[idx2 - 1];
        x2r = a[idx1 - 2] + a[idx3 - 2];
        x2i = a[idx1 - 1] + a[idx3 - 1];
        x3r = a[idx1 - 2] - a[idx3 - 2];
        x3i = a[idx1 - 1] - a[idx3 - 1];
        a[idx0 - 2] = x0r + x2r;
        a[idx0 - 1] = x0i + x2i;
        a[idx1 - 2] = x0r - x2r;
        a[idx1 - 1] = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a[idx2 - 2] = wk1r * x0r - wk1i * x0i;
        a[idx2 - 1] = wk1r * x0i + wk1i * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a[idx3 - 2] = wk3r * x0r + wk3i * x0i;
        a[idx3 - 1] = wk3r * x0i - wk3i * x0r;
        x0r = a[idx0] + a[idx2];
        x0i = a[idx0 + 1] + a[idx2 + 1];
        x1r = a[idx0] - a[idx2];
        x1i = a[idx0 + 1] - a[idx2 + 1];
        x2r = a[idx1] + a[idx3];
        x2i = a[idx1 + 1] + a[idx3 + 1];
        x3r = a[idx1] - a[idx3];
        x3i = a[idx1 + 1] - a[idx3 + 1];
        a[idx0] = x0r + x2r;
        a[idx0 + 1] = x0i + x2i;
        a[idx1] = x0r - x2r;
        a[idx1 + 1] = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a[idx2] = wn4r * (x0r - x0i);
        a[idx2 + 1] = wn4r * (x0i + x0r);
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a[idx3] = -wn4r * (x0r + x0i);
        a[idx3 + 1] = -wn4r * (x0i - x0r);
        x0r = a[idx0 + 2] + a[idx2 + 2];
        x0i = a[idx0 + 3] + a[idx2 + 3];
        x1r = a[idx0 + 2] - a[idx2 + 2];
        x1i = a[idx0 + 3] - a[idx2 + 3];
        x2r = a[idx1 + 2] + a[idx3 + 2];
        x2i = a[idx1 + 3] + a[idx3 + 3];
        x3r = a[idx1 + 2] - a[idx3 + 2];
        x3i = a[idx1 + 3] - a[idx3 + 3];
        a[idx0 + 2] = x0r + x2r;
        a[idx0 + 3] = x0i + x2i;
        a[idx1 + 2] = x0r - x2r;
        a[idx1 + 3] = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a[idx2 + 2] = wk1i * x0r - wk1r * x0i;
        a[idx2 + 3] = wk1i * x0i + wk1r * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a[idx3 + 2] = wk3i * x0r + wk3r * x0i;
        a[idx3 + 3] = wk3i * x0i - wk3r * x0r;
    }

    public static void cftf1st(long n, FloatLargeArray a, long offa, FloatLargeArray w, long startw)
    {
        long j0, j1, j2, j3, k, m, mh;
        float wn4r, csc1, csc3, wk1r, wk1i, wk3r, wk3i, wd1r, wd1i, wd3r, wd3i;
        float x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        long idx0, idx1, idx2, idx3, idx4, idx5;
        mh = n >> 3l;
        m = 2 * mh;
        j1 = m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a.getFloat(offa) + a.getFloat(idx2);
        x0i = a.getFloat(offa + 1) + a.getFloat(idx2 + 1);
        x1r = a.getFloat(offa) - a.getFloat(idx2);
        x1i = a.getFloat(offa + 1) - a.getFloat(idx2 + 1);
        x2r = a.getFloat(idx1) + a.getFloat(idx3);
        x2i = a.getFloat(idx1 + 1) + a.getFloat(idx3 + 1);
        x3r = a.getFloat(idx1) - a.getFloat(idx3);
        x3i = a.getFloat(idx1 + 1) - a.getFloat(idx3 + 1);
        a.setFloat(offa, x0r + x2r);
        a.setFloat(offa + 1, x0i + x2i);
        a.setFloat(idx1, x0r - x2r);
        a.setFloat(idx1 + 1, x0i - x2i);
        a.setFloat(idx2, x1r - x3i);
        a.setFloat(idx2 + 1, x1i + x3r);
        a.setFloat(idx3, x1r + x3i);
        a.setFloat(idx3 + 1, x1i - x3r);
        wn4r = w.getFloat(startw + 1);
        csc1 = w.getFloat(startw + 2);
        csc3 = w.getFloat(startw + 3);
        wd1r = 1;
        wd1i = 0;
        wd3r = 1;
        wd3i = 0;
        k = 0;
        for (int j = 2; j < mh - 2; j += 4) {
            k += 4;
            idx4 = startw + k;
            wk1r = csc1 * (wd1r + w.getFloat(idx4));
            wk1i = csc1 * (wd1i + w.getFloat(idx4 + 1));
            wk3r = csc3 * (wd3r + w.getFloat(idx4 + 2));
            wk3i = csc3 * (wd3i + w.getFloat(idx4 + 3));
            wd1r = w.getFloat(idx4);
            wd1i = w.getFloat(idx4 + 1);
            wd3r = w.getFloat(idx4 + 2);
            wd3i = w.getFloat(idx4 + 3);
            j1 = j + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            idx5 = offa + j;
            x0r = a.getFloat(idx5) + a.getFloat(idx2);
            x0i = a.getFloat(idx5 + 1) + a.getFloat(idx2 + 1);
            x1r = a.getFloat(idx5) - a.getFloat(idx2);
            x1i = a.getFloat(idx5 + 1) - a.getFloat(idx2 + 1);
            y0r = a.getFloat(idx5 + 2) + a.getFloat(idx2 + 2);
            y0i = a.getFloat(idx5 + 3) + a.getFloat(idx2 + 3);
            y1r = a.getFloat(idx5 + 2) - a.getFloat(idx2 + 2);
            y1i = a.getFloat(idx5 + 3) - a.getFloat(idx2 + 3);
            x2r = a.getFloat(idx1) + a.getFloat(idx3);
            x2i = a.getFloat(idx1 + 1) + a.getFloat(idx3 + 1);
            x3r = a.getFloat(idx1) - a.getFloat(idx3);
            x3i = a.getFloat(idx1 + 1) - a.getFloat(idx3 + 1);
            y2r = a.getFloat(idx1 + 2) + a.getFloat(idx3 + 2);
            y2i = a.getFloat(idx1 + 3) + a.getFloat(idx3 + 3);
            y3r = a.getFloat(idx1 + 2) - a.getFloat(idx3 + 2);
            y3i = a.getFloat(idx1 + 3) - a.getFloat(idx3 + 3);
            a.setFloat(idx5, x0r + x2r);
            a.setFloat(idx5 + 1, x0i + x2i);
            a.setFloat(idx5 + 2, y0r + y2r);
            a.setFloat(idx5 + 3, y0i + y2i);
            a.setFloat(idx1, x0r - x2r);
            a.setFloat(idx1 + 1, x0i - x2i);
            a.setFloat(idx1 + 2, y0r - y2r);
            a.setFloat(idx1 + 3, y0i - y2i);
            x0r = x1r - x3i;
            x0i = x1i + x3r;
            a.setFloat(idx2, wk1r * x0r - wk1i * x0i);
            a.setFloat(idx2 + 1, wk1r * x0i + wk1i * x0r);
            x0r = y1r - y3i;
            x0i = y1i + y3r;
            a.setFloat(idx2 + 2, wd1r * x0r - wd1i * x0i);
            a.setFloat(idx2 + 3, wd1r * x0i + wd1i * x0r);
            x0r = x1r + x3i;
            x0i = x1i - x3r;
            a.setFloat(idx3, wk3r * x0r + wk3i * x0i);
            a.setFloat(idx3 + 1, wk3r * x0i - wk3i * x0r);
            x0r = y1r + y3i;
            x0i = y1i - y3r;
            a.setFloat(idx3 + 2, wd3r * x0r + wd3i * x0i);
            a.setFloat(idx3 + 3, wd3r * x0i - wd3i * x0r);
            j0 = m - j;
            j1 = j0 + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx0 = offa + j0;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            x0r = a.getFloat(idx0) + a.getFloat(idx2);
            x0i = a.getFloat(idx0 + 1) + a.getFloat(idx2 + 1);
            x1r = a.getFloat(idx0) - a.getFloat(idx2);
            x1i = a.getFloat(idx0 + 1) - a.getFloat(idx2 + 1);
            y0r = a.getFloat(idx0 - 2) + a.getFloat(idx2 - 2);
            y0i = a.getFloat(idx0 - 1) + a.getFloat(idx2 - 1);
            y1r = a.getFloat(idx0 - 2) - a.getFloat(idx2 - 2);
            y1i = a.getFloat(idx0 - 1) - a.getFloat(idx2 - 1);
            x2r = a.getFloat(idx1) + a.getFloat(idx3);
            x2i = a.getFloat(idx1 + 1) + a.getFloat(idx3 + 1);
            x3r = a.getFloat(idx1) - a.getFloat(idx3);
            x3i = a.getFloat(idx1 + 1) - a.getFloat(idx3 + 1);
            y2r = a.getFloat(idx1 - 2) + a.getFloat(idx3 - 2);
            y2i = a.getFloat(idx1 - 1) + a.getFloat(idx3 - 1);
            y3r = a.getFloat(idx1 - 2) - a.getFloat(idx3 - 2);
            y3i = a.getFloat(idx1 - 1) - a.getFloat(idx3 - 1);
            a.setFloat(idx0, x0r + x2r);
            a.setFloat(idx0 + 1, x0i + x2i);
            a.setFloat(idx0 - 2, y0r + y2r);
            a.setFloat(idx0 - 1, y0i + y2i);
            a.setFloat(idx1, x0r - x2r);
            a.setFloat(idx1 + 1, x0i - x2i);
            a.setFloat(idx1 - 2, y0r - y2r);
            a.setFloat(idx1 - 1, y0i - y2i);
            x0r = x1r - x3i;
            x0i = x1i + x3r;
            a.setFloat(idx2, wk1i * x0r - wk1r * x0i);
            a.setFloat(idx2 + 1, wk1i * x0i + wk1r * x0r);
            x0r = y1r - y3i;
            x0i = y1i + y3r;
            a.setFloat(idx2 - 2, wd1i * x0r - wd1r * x0i);
            a.setFloat(idx2 - 1, wd1i * x0i + wd1r * x0r);
            x0r = x1r + x3i;
            x0i = x1i - x3r;
            a.setFloat(idx3, wk3i * x0r + wk3r * x0i);
            a.setFloat(idx3 + 1, wk3i * x0i - wk3r * x0r);
            x0r = y1r + y3i;
            x0i = y1i - y3r;
            a.setFloat(offa + j3 - 2, wd3i * x0r + wd3r * x0i);
            a.setFloat(offa + j3 - 1, wd3i * x0i - wd3r * x0r);
        }
        wk1r = csc1 * (wd1r + wn4r);
        wk1i = csc1 * (wd1i + wn4r);
        wk3r = csc3 * (wd3r - wn4r);
        wk3i = csc3 * (wd3i - wn4r);
        j0 = mh;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx0 = offa + j0;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a.getFloat(idx0 - 2) + a.getFloat(idx2 - 2);
        x0i = a.getFloat(idx0 - 1) + a.getFloat(idx2 - 1);
        x1r = a.getFloat(idx0 - 2) - a.getFloat(idx2 - 2);
        x1i = a.getFloat(idx0 - 1) - a.getFloat(idx2 - 1);
        x2r = a.getFloat(idx1 - 2) + a.getFloat(idx3 - 2);
        x2i = a.getFloat(idx1 - 1) + a.getFloat(idx3 - 1);
        x3r = a.getFloat(idx1 - 2) - a.getFloat(idx3 - 2);
        x3i = a.getFloat(idx1 - 1) - a.getFloat(idx3 - 1);
        a.setFloat(idx0 - 2, x0r + x2r);
        a.setFloat(idx0 - 1, x0i + x2i);
        a.setFloat(idx1 - 2, x0r - x2r);
        a.setFloat(idx1 - 1, x0i - x2i);
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a.setFloat(idx2 - 2, wk1r * x0r - wk1i * x0i);
        a.setFloat(idx2 - 1, wk1r * x0i + wk1i * x0r);
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a.setFloat(idx3 - 2, wk3r * x0r + wk3i * x0i);
        a.setFloat(idx3 - 1, wk3r * x0i - wk3i * x0r);
        x0r = a.getFloat(idx0) + a.getFloat(idx2);
        x0i = a.getFloat(idx0 + 1) + a.getFloat(idx2 + 1);
        x1r = a.getFloat(idx0) - a.getFloat(idx2);
        x1i = a.getFloat(idx0 + 1) - a.getFloat(idx2 + 1);
        x2r = a.getFloat(idx1) + a.getFloat(idx3);
        x2i = a.getFloat(idx1 + 1) + a.getFloat(idx3 + 1);
        x3r = a.getFloat(idx1) - a.getFloat(idx3);
        x3i = a.getFloat(idx1 + 1) - a.getFloat(idx3 + 1);
        a.setFloat(idx0, x0r + x2r);
        a.setFloat(idx0 + 1, x0i + x2i);
        a.setFloat(idx1, x0r - x2r);
        a.setFloat(idx1 + 1, x0i - x2i);
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a.setFloat(idx2, wn4r * (x0r - x0i));
        a.setFloat(idx2 + 1, wn4r * (x0i + x0r));
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a.setFloat(idx3, -wn4r * (x0r + x0i));
        a.setFloat(idx3 + 1, -wn4r * (x0i - x0r));
        x0r = a.getFloat(idx0 + 2) + a.getFloat(idx2 + 2);
        x0i = a.getFloat(idx0 + 3) + a.getFloat(idx2 + 3);
        x1r = a.getFloat(idx0 + 2) - a.getFloat(idx2 + 2);
        x1i = a.getFloat(idx0 + 3) - a.getFloat(idx2 + 3);
        x2r = a.getFloat(idx1 + 2) + a.getFloat(idx3 + 2);
        x2i = a.getFloat(idx1 + 3) + a.getFloat(idx3 + 3);
        x3r = a.getFloat(idx1 + 2) - a.getFloat(idx3 + 2);
        x3i = a.getFloat(idx1 + 3) - a.getFloat(idx3 + 3);
        a.setFloat(idx0 + 2, x0r + x2r);
        a.setFloat(idx0 + 3, x0i + x2i);
        a.setFloat(idx1 + 2, x0r - x2r);
        a.setFloat(idx1 + 3, x0i - x2i);
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a.setFloat(idx2 + 2, wk1i * x0r - wk1r * x0i);
        a.setFloat(idx2 + 3, wk1i * x0i + wk1r * x0r);
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a.setFloat(idx3 + 2, wk3i * x0r + wk3r * x0i);
        a.setFloat(idx3 + 3, wk3i * x0i - wk3r * x0r);
    }

    public static void cftb1st(int n, float[] a, int offa, float[] w, int startw)
    {
        int j0, j1, j2, j3, k, m, mh;
        float wn4r, csc1, csc3, wk1r, wk1i, wk3r, wk3i, wd1r, wd1i, wd3r, wd3i;
        float x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        int idx0, idx1, idx2, idx3, idx4, idx5;
        mh = n >> 3;
        m = 2 * mh;
        j1 = m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;

        x0r = a[offa] + a[idx2];
        x0i = -a[offa + 1] - a[idx2 + 1];
        x1r = a[offa] - a[idx2];
        x1i = -a[offa + 1] + a[idx2 + 1];
        x2r = a[idx1] + a[idx3];
        x2i = a[idx1 + 1] + a[idx3 + 1];
        x3r = a[idx1] - a[idx3];
        x3i = a[idx1 + 1] - a[idx3 + 1];
        a[offa] = x0r + x2r;
        a[offa + 1] = x0i - x2i;
        a[idx1] = x0r - x2r;
        a[idx1 + 1] = x0i + x2i;
        a[idx2] = x1r + x3i;
        a[idx2 + 1] = x1i + x3r;
        a[idx3] = x1r - x3i;
        a[idx3 + 1] = x1i - x3r;
        wn4r = w[startw + 1];
        csc1 = w[startw + 2];
        csc3 = w[startw + 3];
        wd1r = 1;
        wd1i = 0;
        wd3r = 1;
        wd3i = 0;
        k = 0;
        for (int j = 2; j < mh - 2; j += 4) {
            k += 4;
            idx4 = startw + k;
            wk1r = csc1 * (wd1r + w[idx4]);
            wk1i = csc1 * (wd1i + w[idx4 + 1]);
            wk3r = csc3 * (wd3r + w[idx4 + 2]);
            wk3i = csc3 * (wd3i + w[idx4 + 3]);
            wd1r = w[idx4];
            wd1i = w[idx4 + 1];
            wd3r = w[idx4 + 2];
            wd3i = w[idx4 + 3];
            j1 = j + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            idx5 = offa + j;
            x0r = a[idx5] + a[idx2];
            x0i = -a[idx5 + 1] - a[idx2 + 1];
            x1r = a[idx5] - a[offa + j2];
            x1i = -a[idx5 + 1] + a[idx2 + 1];
            y0r = a[idx5 + 2] + a[idx2 + 2];
            y0i = -a[idx5 + 3] - a[idx2 + 3];
            y1r = a[idx5 + 2] - a[idx2 + 2];
            y1i = -a[idx5 + 3] + a[idx2 + 3];
            x2r = a[idx1] + a[idx3];
            x2i = a[idx1 + 1] + a[idx3 + 1];
            x3r = a[idx1] - a[idx3];
            x3i = a[idx1 + 1] - a[idx3 + 1];
            y2r = a[idx1 + 2] + a[idx3 + 2];
            y2i = a[idx1 + 3] + a[idx3 + 3];
            y3r = a[idx1 + 2] - a[idx3 + 2];
            y3i = a[idx1 + 3] - a[idx3 + 3];
            a[idx5] = x0r + x2r;
            a[idx5 + 1] = x0i - x2i;
            a[idx5 + 2] = y0r + y2r;
            a[idx5 + 3] = y0i - y2i;
            a[idx1] = x0r - x2r;
            a[idx1 + 1] = x0i + x2i;
            a[idx1 + 2] = y0r - y2r;
            a[idx1 + 3] = y0i + y2i;
            x0r = x1r + x3i;
            x0i = x1i + x3r;
            a[idx2] = wk1r * x0r - wk1i * x0i;
            a[idx2 + 1] = wk1r * x0i + wk1i * x0r;
            x0r = y1r + y3i;
            x0i = y1i + y3r;
            a[idx2 + 2] = wd1r * x0r - wd1i * x0i;
            a[idx2 + 3] = wd1r * x0i + wd1i * x0r;
            x0r = x1r - x3i;
            x0i = x1i - x3r;
            a[idx3] = wk3r * x0r + wk3i * x0i;
            a[idx3 + 1] = wk3r * x0i - wk3i * x0r;
            x0r = y1r - y3i;
            x0i = y1i - y3r;
            a[idx3 + 2] = wd3r * x0r + wd3i * x0i;
            a[idx3 + 3] = wd3r * x0i - wd3i * x0r;
            j0 = m - j;
            j1 = j0 + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx0 = offa + j0;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            x0r = a[idx0] + a[idx2];
            x0i = -a[idx0 + 1] - a[idx2 + 1];
            x1r = a[idx0] - a[idx2];
            x1i = -a[idx0 + 1] + a[idx2 + 1];
            y0r = a[idx0 - 2] + a[idx2 - 2];
            y0i = -a[idx0 - 1] - a[idx2 - 1];
            y1r = a[idx0 - 2] - a[idx2 - 2];
            y1i = -a[idx0 - 1] + a[idx2 - 1];
            x2r = a[idx1] + a[idx3];
            x2i = a[idx1 + 1] + a[idx3 + 1];
            x3r = a[idx1] - a[idx3];
            x3i = a[idx1 + 1] - a[idx3 + 1];
            y2r = a[idx1 - 2] + a[idx3 - 2];
            y2i = a[idx1 - 1] + a[idx3 - 1];
            y3r = a[idx1 - 2] - a[idx3 - 2];
            y3i = a[idx1 - 1] - a[idx3 - 1];
            a[idx0] = x0r + x2r;
            a[idx0 + 1] = x0i - x2i;
            a[idx0 - 2] = y0r + y2r;
            a[idx0 - 1] = y0i - y2i;
            a[idx1] = x0r - x2r;
            a[idx1 + 1] = x0i + x2i;
            a[idx1 - 2] = y0r - y2r;
            a[idx1 - 1] = y0i + y2i;
            x0r = x1r + x3i;
            x0i = x1i + x3r;
            a[idx2] = wk1i * x0r - wk1r * x0i;
            a[idx2 + 1] = wk1i * x0i + wk1r * x0r;
            x0r = y1r + y3i;
            x0i = y1i + y3r;
            a[idx2 - 2] = wd1i * x0r - wd1r * x0i;
            a[idx2 - 1] = wd1i * x0i + wd1r * x0r;
            x0r = x1r - x3i;
            x0i = x1i - x3r;
            a[idx3] = wk3i * x0r + wk3r * x0i;
            a[idx3 + 1] = wk3i * x0i - wk3r * x0r;
            x0r = y1r - y3i;
            x0i = y1i - y3r;
            a[idx3 - 2] = wd3i * x0r + wd3r * x0i;
            a[idx3 - 1] = wd3i * x0i - wd3r * x0r;
        }
        wk1r = csc1 * (wd1r + wn4r);
        wk1i = csc1 * (wd1i + wn4r);
        wk3r = csc3 * (wd3r - wn4r);
        wk3i = csc3 * (wd3i - wn4r);
        j0 = mh;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx0 = offa + j0;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a[idx0 - 2] + a[idx2 - 2];
        x0i = -a[idx0 - 1] - a[idx2 - 1];
        x1r = a[idx0 - 2] - a[idx2 - 2];
        x1i = -a[idx0 - 1] + a[idx2 - 1];
        x2r = a[idx1 - 2] + a[idx3 - 2];
        x2i = a[idx1 - 1] + a[idx3 - 1];
        x3r = a[idx1 - 2] - a[idx3 - 2];
        x3i = a[idx1 - 1] - a[idx3 - 1];
        a[idx0 - 2] = x0r + x2r;
        a[idx0 - 1] = x0i - x2i;
        a[idx1 - 2] = x0r - x2r;
        a[idx1 - 1] = x0i + x2i;
        x0r = x1r + x3i;
        x0i = x1i + x3r;
        a[idx2 - 2] = wk1r * x0r - wk1i * x0i;
        a[idx2 - 1] = wk1r * x0i + wk1i * x0r;
        x0r = x1r - x3i;
        x0i = x1i - x3r;
        a[idx3 - 2] = wk3r * x0r + wk3i * x0i;
        a[idx3 - 1] = wk3r * x0i - wk3i * x0r;
        x0r = a[idx0] + a[idx2];
        x0i = -a[idx0 + 1] - a[idx2 + 1];
        x1r = a[idx0] - a[idx2];
        x1i = -a[idx0 + 1] + a[idx2 + 1];
        x2r = a[idx1] + a[idx3];
        x2i = a[idx1 + 1] + a[idx3 + 1];
        x3r = a[idx1] - a[idx3];
        x3i = a[idx1 + 1] - a[idx3 + 1];
        a[idx0] = x0r + x2r;
        a[idx0 + 1] = x0i - x2i;
        a[idx1] = x0r - x2r;
        a[idx1 + 1] = x0i + x2i;
        x0r = x1r + x3i;
        x0i = x1i + x3r;
        a[idx2] = wn4r * (x0r - x0i);
        a[idx2 + 1] = wn4r * (x0i + x0r);
        x0r = x1r - x3i;
        x0i = x1i - x3r;
        a[idx3] = -wn4r * (x0r + x0i);
        a[idx3 + 1] = -wn4r * (x0i - x0r);
        x0r = a[idx0 + 2] + a[idx2 + 2];
        x0i = -a[idx0 + 3] - a[idx2 + 3];
        x1r = a[idx0 + 2] - a[idx2 + 2];
        x1i = -a[idx0 + 3] + a[idx2 + 3];
        x2r = a[idx1 + 2] + a[idx3 + 2];
        x2i = a[idx1 + 3] + a[idx3 + 3];
        x3r = a[idx1 + 2] - a[idx3 + 2];
        x3i = a[idx1 + 3] - a[idx3 + 3];
        a[idx0 + 2] = x0r + x2r;
        a[idx0 + 3] = x0i - x2i;
        a[idx1 + 2] = x0r - x2r;
        a[idx1 + 3] = x0i + x2i;
        x0r = x1r + x3i;
        x0i = x1i + x3r;
        a[idx2 + 2] = wk1i * x0r - wk1r * x0i;
        a[idx2 + 3] = wk1i * x0i + wk1r * x0r;
        x0r = x1r - x3i;
        x0i = x1i - x3r;
        a[idx3 + 2] = wk3i * x0r + wk3r * x0i;
        a[idx3 + 3] = wk3i * x0i - wk3r * x0r;
    }

    public static void cftb1st(long n, FloatLargeArray a, long offa, FloatLargeArray w, long startw)
    {
        long j0, j1, j2, j3, k, m, mh;
        float wn4r, csc1, csc3, wk1r, wk1i, wk3r, wk3i, wd1r, wd1i, wd3r, wd3i;
        float x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        long idx0, idx1, idx2, idx3, idx4, idx5;
        mh = n >> 3l;
        m = 2 * mh;
        j1 = m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;

        x0r = a.getFloat(offa) + a.getFloat(idx2);
        x0i = -a.getFloat(offa + 1) - a.getFloat(idx2 + 1);
        x1r = a.getFloat(offa) - a.getFloat(idx2);
        x1i = -a.getFloat(offa + 1) + a.getFloat(idx2 + 1);
        x2r = a.getFloat(idx1) + a.getFloat(idx3);
        x2i = a.getFloat(idx1 + 1) + a.getFloat(idx3 + 1);
        x3r = a.getFloat(idx1) - a.getFloat(idx3);
        x3i = a.getFloat(idx1 + 1) - a.getFloat(idx3 + 1);
        a.setFloat(offa, x0r + x2r);
        a.setFloat(offa + 1, x0i - x2i);
        a.setFloat(idx1, x0r - x2r);
        a.setFloat(idx1 + 1, x0i + x2i);
        a.setFloat(idx2, x1r + x3i);
        a.setFloat(idx2 + 1, x1i + x3r);
        a.setFloat(idx3, x1r - x3i);
        a.setFloat(idx3 + 1, x1i - x3r);
        wn4r = w.getFloat(startw + 1);
        csc1 = w.getFloat(startw + 2);
        csc3 = w.getFloat(startw + 3);
        wd1r = 1;
        wd1i = 0;
        wd3r = 1;
        wd3i = 0;
        k = 0;
        for (long j = 2; j < mh - 2; j += 4) {
            k += 4;
            idx4 = startw + k;
            wk1r = csc1 * (wd1r + w.getFloat(idx4));
            wk1i = csc1 * (wd1i + w.getFloat(idx4 + 1));
            wk3r = csc3 * (wd3r + w.getFloat(idx4 + 2));
            wk3i = csc3 * (wd3i + w.getFloat(idx4 + 3));
            wd1r = w.getFloat(idx4);
            wd1i = w.getFloat(idx4 + 1);
            wd3r = w.getFloat(idx4 + 2);
            wd3i = w.getFloat(idx4 + 3);
            j1 = j + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            idx5 = offa + j;
            x0r = a.getFloat(idx5) + a.getFloat(idx2);
            x0i = -a.getFloat(idx5 + 1) - a.getFloat(idx2 + 1);
            x1r = a.getFloat(idx5) - a.getFloat(offa + j2);
            x1i = -a.getFloat(idx5 + 1) + a.getFloat(idx2 + 1);
            y0r = a.getFloat(idx5 + 2) + a.getFloat(idx2 + 2);
            y0i = -a.getFloat(idx5 + 3) - a.getFloat(idx2 + 3);
            y1r = a.getFloat(idx5 + 2) - a.getFloat(idx2 + 2);
            y1i = -a.getFloat(idx5 + 3) + a.getFloat(idx2 + 3);
            x2r = a.getFloat(idx1) + a.getFloat(idx3);
            x2i = a.getFloat(idx1 + 1) + a.getFloat(idx3 + 1);
            x3r = a.getFloat(idx1) - a.getFloat(idx3);
            x3i = a.getFloat(idx1 + 1) - a.getFloat(idx3 + 1);
            y2r = a.getFloat(idx1 + 2) + a.getFloat(idx3 + 2);
            y2i = a.getFloat(idx1 + 3) + a.getFloat(idx3 + 3);
            y3r = a.getFloat(idx1 + 2) - a.getFloat(idx3 + 2);
            y3i = a.getFloat(idx1 + 3) - a.getFloat(idx3 + 3);
            a.setFloat(idx5, x0r + x2r);
            a.setFloat(idx5 + 1, x0i - x2i);
            a.setFloat(idx5 + 2, y0r + y2r);
            a.setFloat(idx5 + 3, y0i - y2i);
            a.setFloat(idx1, x0r - x2r);
            a.setFloat(idx1 + 1, x0i + x2i);
            a.setFloat(idx1 + 2, y0r - y2r);
            a.setFloat(idx1 + 3, y0i + y2i);
            x0r = x1r + x3i;
            x0i = x1i + x3r;
            a.setFloat(idx2, wk1r * x0r - wk1i * x0i);
            a.setFloat(idx2 + 1, wk1r * x0i + wk1i * x0r);
            x0r = y1r + y3i;
            x0i = y1i + y3r;
            a.setFloat(idx2 + 2, wd1r * x0r - wd1i * x0i);
            a.setFloat(idx2 + 3, wd1r * x0i + wd1i * x0r);
            x0r = x1r - x3i;
            x0i = x1i - x3r;
            a.setFloat(idx3, wk3r * x0r + wk3i * x0i);
            a.setFloat(idx3 + 1, wk3r * x0i - wk3i * x0r);
            x0r = y1r - y3i;
            x0i = y1i - y3r;
            a.setFloat(idx3 + 2, wd3r * x0r + wd3i * x0i);
            a.setFloat(idx3 + 3, wd3r * x0i - wd3i * x0r);
            j0 = m - j;
            j1 = j0 + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx0 = offa + j0;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            x0r = a.getFloat(idx0) + a.getFloat(idx2);
            x0i = -a.getFloat(idx0 + 1) - a.getFloat(idx2 + 1);
            x1r = a.getFloat(idx0) - a.getFloat(idx2);
            x1i = -a.getFloat(idx0 + 1) + a.getFloat(idx2 + 1);
            y0r = a.getFloat(idx0 - 2) + a.getFloat(idx2 - 2);
            y0i = -a.getFloat(idx0 - 1) - a.getFloat(idx2 - 1);
            y1r = a.getFloat(idx0 - 2) - a.getFloat(idx2 - 2);
            y1i = -a.getFloat(idx0 - 1) + a.getFloat(idx2 - 1);
            x2r = a.getFloat(idx1) + a.getFloat(idx3);
            x2i = a.getFloat(idx1 + 1) + a.getFloat(idx3 + 1);
            x3r = a.getFloat(idx1) - a.getFloat(idx3);
            x3i = a.getFloat(idx1 + 1) - a.getFloat(idx3 + 1);
            y2r = a.getFloat(idx1 - 2) + a.getFloat(idx3 - 2);
            y2i = a.getFloat(idx1 - 1) + a.getFloat(idx3 - 1);
            y3r = a.getFloat(idx1 - 2) - a.getFloat(idx3 - 2);
            y3i = a.getFloat(idx1 - 1) - a.getFloat(idx3 - 1);
            a.setFloat(idx0, x0r + x2r);
            a.setFloat(idx0 + 1, x0i - x2i);
            a.setFloat(idx0 - 2, y0r + y2r);
            a.setFloat(idx0 - 1, y0i - y2i);
            a.setFloat(idx1, x0r - x2r);
            a.setFloat(idx1 + 1, x0i + x2i);
            a.setFloat(idx1 - 2, y0r - y2r);
            a.setFloat(idx1 - 1, y0i + y2i);
            x0r = x1r + x3i;
            x0i = x1i + x3r;
            a.setFloat(idx2, wk1i * x0r - wk1r * x0i);
            a.setFloat(idx2 + 1, wk1i * x0i + wk1r * x0r);
            x0r = y1r + y3i;
            x0i = y1i + y3r;
            a.setFloat(idx2 - 2, wd1i * x0r - wd1r * x0i);
            a.setFloat(idx2 - 1, wd1i * x0i + wd1r * x0r);
            x0r = x1r - x3i;
            x0i = x1i - x3r;
            a.setFloat(idx3, wk3i * x0r + wk3r * x0i);
            a.setFloat(idx3 + 1, wk3i * x0i - wk3r * x0r);
            x0r = y1r - y3i;
            x0i = y1i - y3r;
            a.setFloat(idx3 - 2, wd3i * x0r + wd3r * x0i);
            a.setFloat(idx3 - 1, wd3i * x0i - wd3r * x0r);
        }
        wk1r = csc1 * (wd1r + wn4r);
        wk1i = csc1 * (wd1i + wn4r);
        wk3r = csc3 * (wd3r - wn4r);
        wk3i = csc3 * (wd3i - wn4r);
        j0 = mh;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx0 = offa + j0;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a.getFloat(idx0 - 2) + a.getFloat(idx2 - 2);
        x0i = -a.getFloat(idx0 - 1) - a.getFloat(idx2 - 1);
        x1r = a.getFloat(idx0 - 2) - a.getFloat(idx2 - 2);
        x1i = -a.getFloat(idx0 - 1) + a.getFloat(idx2 - 1);
        x2r = a.getFloat(idx1 - 2) + a.getFloat(idx3 - 2);
        x2i = a.getFloat(idx1 - 1) + a.getFloat(idx3 - 1);
        x3r = a.getFloat(idx1 - 2) - a.getFloat(idx3 - 2);
        x3i = a.getFloat(idx1 - 1) - a.getFloat(idx3 - 1);
        a.setFloat(idx0 - 2, x0r + x2r);
        a.setFloat(idx0 - 1, x0i - x2i);
        a.setFloat(idx1 - 2, x0r - x2r);
        a.setFloat(idx1 - 1, x0i + x2i);
        x0r = x1r + x3i;
        x0i = x1i + x3r;
        a.setFloat(idx2 - 2, wk1r * x0r - wk1i * x0i);
        a.setFloat(idx2 - 1, wk1r * x0i + wk1i * x0r);
        x0r = x1r - x3i;
        x0i = x1i - x3r;
        a.setFloat(idx3 - 2, wk3r * x0r + wk3i * x0i);
        a.setFloat(idx3 - 1, wk3r * x0i - wk3i * x0r);
        x0r = a.getFloat(idx0) + a.getFloat(idx2);
        x0i = -a.getFloat(idx0 + 1) - a.getFloat(idx2 + 1);
        x1r = a.getFloat(idx0) - a.getFloat(idx2);
        x1i = -a.getFloat(idx0 + 1) + a.getFloat(idx2 + 1);
        x2r = a.getFloat(idx1) + a.getFloat(idx3);
        x2i = a.getFloat(idx1 + 1) + a.getFloat(idx3 + 1);
        x3r = a.getFloat(idx1) - a.getFloat(idx3);
        x3i = a.getFloat(idx1 + 1) - a.getFloat(idx3 + 1);
        a.setFloat(idx0, x0r + x2r);
        a.setFloat(idx0 + 1, x0i - x2i);
        a.setFloat(idx1, x0r - x2r);
        a.setFloat(idx1 + 1, x0i + x2i);
        x0r = x1r + x3i;
        x0i = x1i + x3r;
        a.setFloat(idx2, wn4r * (x0r - x0i));
        a.setFloat(idx2 + 1, wn4r * (x0i + x0r));
        x0r = x1r - x3i;
        x0i = x1i - x3r;
        a.setFloat(idx3, -wn4r * (x0r + x0i));
        a.setFloat(idx3 + 1, -wn4r * (x0i - x0r));
        x0r = a.getFloat(idx0 + 2) + a.getFloat(idx2 + 2);
        x0i = -a.getFloat(idx0 + 3) - a.getFloat(idx2 + 3);
        x1r = a.getFloat(idx0 + 2) - a.getFloat(idx2 + 2);
        x1i = -a.getFloat(idx0 + 3) + a.getFloat(idx2 + 3);
        x2r = a.getFloat(idx1 + 2) + a.getFloat(idx3 + 2);
        x2i = a.getFloat(idx1 + 3) + a.getFloat(idx3 + 3);
        x3r = a.getFloat(idx1 + 2) - a.getFloat(idx3 + 2);
        x3i = a.getFloat(idx1 + 3) - a.getFloat(idx3 + 3);
        a.setFloat(idx0 + 2, x0r + x2r);
        a.setFloat(idx0 + 3, x0i - x2i);
        a.setFloat(idx1 + 2, x0r - x2r);
        a.setFloat(idx1 + 3, x0i + x2i);
        x0r = x1r + x3i;
        x0i = x1i + x3r;
        a.setFloat(idx2 + 2, wk1i * x0r - wk1r * x0i);
        a.setFloat(idx2 + 3, wk1i * x0i + wk1r * x0r);
        x0r = x1r - x3i;
        x0i = x1i - x3r;
        a.setFloat(idx3 + 2, wk3i * x0r + wk3r * x0i);
        a.setFloat(idx3 + 3, wk3i * x0i - wk3r * x0r);
    }

    public static void cftrec4_th(final int n, final float[] a, final int offa, final int nw, final float[] w)
    {
        int i;
        int idiv4, m, nthreads;
        int idx = 0;
        nthreads = 2;
        idiv4 = 0;
        m = n >> 1;
        if (n >= CommonUtils.getThreadsBeginN_1D_FFT_4Threads()) {
            nthreads = 4;
            idiv4 = 1;
            m >>= 1;
        }
        Future<?>[] futures = new Future[nthreads];
        final int mf = m;
        for (i = 0; i < nthreads; i++) {
            final int firstIdx = offa + i * m;
            if (i != idiv4) {
                futures[idx++] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        int isplt, j, k, m;
                        int idx1 = firstIdx + mf;
                        m = n;
                        while (m > 512) {
                            m >>= 2;
                            cftmdl1(m, a, idx1 - m, w, nw - (m >> 1));
                        }
                        cftleaf(m, 1, a, idx1 - m, nw, w);
                        k = 0;
                        int idx2 = firstIdx - m;
                        for (j = mf - m; j > 0; j -= m) {
                            k++;
                            isplt = cfttree(m, j, k, a, firstIdx, nw, w);
                            cftleaf(m, isplt, a, idx2 + j, nw, w);
                        }
                    }
                });
            } else {
                futures[idx++] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        int isplt, j, k, m;
                        int idx1 = firstIdx + mf;
                        k = 1;
                        m = n;
                        while (m > 512) {
                            m >>= 2;
                            k <<= 2;
                            cftmdl2(m, a, idx1 - m, w, nw - m);
                        }
                        cftleaf(m, 0, a, idx1 - m, nw, w);
                        k >>= 1;
                        int idx2 = firstIdx - m;
                        for (j = mf - m; j > 0; j -= m) {
                            k++;
                            isplt = cfttree(m, j, k, a, firstIdx, nw, w);
                            cftleaf(m, isplt, a, idx2 + j, nw, w);
                        }
                    }
                });
            }
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(CommonUtils.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(CommonUtils.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static void cftrec4_th(final long n, final FloatLargeArray a, final long offa, final long nw, final FloatLargeArray w)
    {
        int i, idx = 0;
        int idiv4, nthreads;
        long m;
        nthreads = 2;
        idiv4 = 0;
        m = n >> 1l;
        if (n >= CommonUtils.getThreadsBeginN_1D_FFT_4Threads()) {
            nthreads = 4;
            idiv4 = 1;
            m >>= 1l;
        }
        Future<?>[] futures = new Future[nthreads];
        final long mf = m;
        for (i = 0; i < nthreads; i++) {
            final long firstIdx = offa + i * m;
            if (i != idiv4) {
                futures[idx++] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        long isplt, j, k, m;
                        long idx1 = firstIdx + mf;
                        m = n;
                        while (m > 512) {
                            m >>= 2;
                            cftmdl1(m, a, idx1 - m, w, nw - (m >> 1l));
                        }
                        cftleaf(m, 1, a, idx1 - m, nw, w);
                        k = 0;
                        long idx2 = firstIdx - m;
                        for (j = mf - m; j > 0; j -= m) {
                            k++;
                            isplt = cfttree(m, j, k, a, firstIdx, nw, w);
                            cftleaf(m, isplt, a, idx2 + j, nw, w);
                        }
                    }
                });
            } else {
                futures[idx++] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        long isplt, j, k, m;
                        long idx1 = firstIdx + mf;
                        k = 1;
                        m = n;
                        while (m > 512) {
                            m >>= 2;
                            k <<= 2;
                            cftmdl2(m, a, idx1 - m, w, nw - m);
                        }
                        cftleaf(m, 0, a, idx1 - m, nw, w);
                        k >>= 1l;
                        long idx2 = firstIdx - m;
                        for (j = mf - m; j > 0; j -= m) {
                            k++;
                            isplt = cfttree(m, j, k, a, firstIdx, nw, w);
                            cftleaf(m, isplt, a, idx2 + j, nw, w);
                        }
                    }
                });
            }
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(CommonUtils.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(CommonUtils.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static void cftrec4(int n, float[] a, int offa, int nw, float[] w)
    {
        int isplt, j, k, m;

        m = n;
        int idx1 = offa + n;
        while (m > 512) {
            m >>= 2;
            cftmdl1(m, a, idx1 - m, w, nw - (m >> 1));
        }
        cftleaf(m, 1, a, idx1 - m, nw, w);
        k = 0;
        int idx2 = offa - m;
        for (j = n - m; j > 0; j -= m) {
            k++;
            isplt = cfttree(m, j, k, a, offa, nw, w);
            cftleaf(m, isplt, a, idx2 + j, nw, w);
        }
    }

    public static void cftrec4(long n, FloatLargeArray a, long offa, long nw, FloatLargeArray w)
    {
        long isplt, j, k, m;

        m = n;
        long idx1 = offa + n;
        while (m > 512) {
            m >>= 2;
            cftmdl1(m, a, idx1 - m, w, nw - (m >> 1l));
        }
        cftleaf(m, 1, a, idx1 - m, nw, w);
        k = 0;
        long idx2 = offa - m;
        for (j = n - m; j > 0; j -= m) {
            k++;
            isplt = cfttree(m, j, k, a, offa, nw, w);
            cftleaf(m, isplt, a, idx2 + j, nw, w);
        }
    }

    public static int cfttree(int n, int j, int k, float[] a, int offa, int nw, float[] w)
    {
        int i, isplt, m;
        int idx1 = offa - n;
        if ((k & 3) != 0) {
            isplt = k & 1;
            if (isplt != 0) {
                cftmdl1(n, a, idx1 + j, w, nw - (n >> 1));
            } else {
                cftmdl2(n, a, idx1 + j, w, nw - n);
            }
        } else {
            m = n;
            for (i = k; (i & 3) == 0; i >>= 2) {
                m <<= 2;
            }
            isplt = i & 1;
            int idx2 = offa + j;
            if (isplt != 0) {
                while (m > 128) {
                    cftmdl1(m, a, idx2 - m, w, nw - (m >> 1));
                    m >>= 2;
                }
            } else {
                while (m > 128) {
                    cftmdl2(m, a, idx2 - m, w, nw - m);
                    m >>= 2;
                }
            }
        }
        return isplt;
    }

    public static long cfttree(long n, long j, long k, FloatLargeArray a, long offa, long nw, FloatLargeArray w)
    {
        long i, isplt, m;
        long idx1 = offa - n;
        if ((k & 3) != 0) {
            isplt = k & 1;
            if (isplt != 0) {
                cftmdl1(n, a, idx1 + j, w, nw - (n >> 1l));
            } else {
                cftmdl2(n, a, idx1 + j, w, nw - n);
            }
        } else {
            m = n;
            for (i = k; (i & 3) == 0; i >>= 2l) {
                m <<= 2l;
            }
            isplt = i & 1;
            long idx2 = offa + j;
            if (isplt != 0) {
                while (m > 128) {
                    cftmdl1(m, a, idx2 - m, w, nw - (m >> 1l));
                    m >>= 2l;
                }
            } else {
                while (m > 128) {
                    cftmdl2(m, a, idx2 - m, w, nw - m);
                    m >>= 2l;
                }
            }
        }
        return isplt;
    }

    public static void cftleaf(int n, int isplt, float[] a, int offa, int nw, float[] w)
    {
        if (n == 512) {
            cftmdl1(128, a, offa, w, nw - 64);
            cftf161(a, offa, w, nw - 8);
            cftf162(a, offa + 32, w, nw - 32);
            cftf161(a, offa + 64, w, nw - 8);
            cftf161(a, offa + 96, w, nw - 8);
            cftmdl2(128, a, offa + 128, w, nw - 128);
            cftf161(a, offa + 128, w, nw - 8);
            cftf162(a, offa + 160, w, nw - 32);
            cftf161(a, offa + 192, w, nw - 8);
            cftf162(a, offa + 224, w, nw - 32);
            cftmdl1(128, a, offa + 256, w, nw - 64);
            cftf161(a, offa + 256, w, nw - 8);
            cftf162(a, offa + 288, w, nw - 32);
            cftf161(a, offa + 320, w, nw - 8);
            cftf161(a, offa + 352, w, nw - 8);
            if (isplt != 0) {
                cftmdl1(128, a, offa + 384, w, nw - 64);
                cftf161(a, offa + 480, w, nw - 8);
            } else {
                cftmdl2(128, a, offa + 384, w, nw - 128);
                cftf162(a, offa + 480, w, nw - 32);
            }
            cftf161(a, offa + 384, w, nw - 8);
            cftf162(a, offa + 416, w, nw - 32);
            cftf161(a, offa + 448, w, nw - 8);
        } else {
            cftmdl1(64, a, offa, w, nw - 32);
            cftf081(a, offa, w, nw - 8);
            cftf082(a, offa + 16, w, nw - 8);
            cftf081(a, offa + 32, w, nw - 8);
            cftf081(a, offa + 48, w, nw - 8);
            cftmdl2(64, a, offa + 64, w, nw - 64);
            cftf081(a, offa + 64, w, nw - 8);
            cftf082(a, offa + 80, w, nw - 8);
            cftf081(a, offa + 96, w, nw - 8);
            cftf082(a, offa + 112, w, nw - 8);
            cftmdl1(64, a, offa + 128, w, nw - 32);
            cftf081(a, offa + 128, w, nw - 8);
            cftf082(a, offa + 144, w, nw - 8);
            cftf081(a, offa + 160, w, nw - 8);
            cftf081(a, offa + 176, w, nw - 8);
            if (isplt != 0) {
                cftmdl1(64, a, offa + 192, w, nw - 32);
                cftf081(a, offa + 240, w, nw - 8);
            } else {
                cftmdl2(64, a, offa + 192, w, nw - 64);
                cftf082(a, offa + 240, w, nw - 8);
            }
            cftf081(a, offa + 192, w, nw - 8);
            cftf082(a, offa + 208, w, nw - 8);
            cftf081(a, offa + 224, w, nw - 8);
        }
    }

    public static void cftleaf(long n, long isplt, FloatLargeArray a, long offa, long nw, FloatLargeArray w)
    {
        if (n == 512) {
            cftmdl1(128, a, offa, w, nw - 64);
            cftf161(a, offa, w, nw - 8);
            cftf162(a, offa + 32, w, nw - 32);
            cftf161(a, offa + 64, w, nw - 8);
            cftf161(a, offa + 96, w, nw - 8);
            cftmdl2(128, a, offa + 128, w, nw - 128);
            cftf161(a, offa + 128, w, nw - 8);
            cftf162(a, offa + 160, w, nw - 32);
            cftf161(a, offa + 192, w, nw - 8);
            cftf162(a, offa + 224, w, nw - 32);
            cftmdl1(128, a, offa + 256, w, nw - 64);
            cftf161(a, offa + 256, w, nw - 8);
            cftf162(a, offa + 288, w, nw - 32);
            cftf161(a, offa + 320, w, nw - 8);
            cftf161(a, offa + 352, w, nw - 8);
            if (isplt != 0) {
                cftmdl1(128, a, offa + 384, w, nw - 64);
                cftf161(a, offa + 480, w, nw - 8);
            } else {
                cftmdl2(128, a, offa + 384, w, nw - 128);
                cftf162(a, offa + 480, w, nw - 32);
            }
            cftf161(a, offa + 384, w, nw - 8);
            cftf162(a, offa + 416, w, nw - 32);
            cftf161(a, offa + 448, w, nw - 8);
        } else {
            cftmdl1(64, a, offa, w, nw - 32);
            cftf081(a, offa, w, nw - 8);
            cftf082(a, offa + 16, w, nw - 8);
            cftf081(a, offa + 32, w, nw - 8);
            cftf081(a, offa + 48, w, nw - 8);
            cftmdl2(64, a, offa + 64, w, nw - 64);
            cftf081(a, offa + 64, w, nw - 8);
            cftf082(a, offa + 80, w, nw - 8);
            cftf081(a, offa + 96, w, nw - 8);
            cftf082(a, offa + 112, w, nw - 8);
            cftmdl1(64, a, offa + 128, w, nw - 32);
            cftf081(a, offa + 128, w, nw - 8);
            cftf082(a, offa + 144, w, nw - 8);
            cftf081(a, offa + 160, w, nw - 8);
            cftf081(a, offa + 176, w, nw - 8);
            if (isplt != 0) {
                cftmdl1(64, a, offa + 192, w, nw - 32);
                cftf081(a, offa + 240, w, nw - 8);
            } else {
                cftmdl2(64, a, offa + 192, w, nw - 64);
                cftf082(a, offa + 240, w, nw - 8);
            }
            cftf081(a, offa + 192, w, nw - 8);
            cftf082(a, offa + 208, w, nw - 8);
            cftf081(a, offa + 224, w, nw - 8);
        }
    }

    public static void cftmdl1(int n, float[] a, int offa, float[] w, int startw)
    {
        int j0, j1, j2, j3, k, m, mh;
        float wn4r, wk1r, wk1i, wk3r, wk3i;
        float x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;
        int idx0, idx1, idx2, idx3, idx4, idx5;

        mh = n >> 3;
        m = 2 * mh;
        j1 = m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a[offa] + a[idx2];
        x0i = a[offa + 1] + a[idx2 + 1];
        x1r = a[offa] - a[idx2];
        x1i = a[offa + 1] - a[idx2 + 1];
        x2r = a[idx1] + a[idx3];
        x2i = a[idx1 + 1] + a[idx3 + 1];
        x3r = a[idx1] - a[idx3];
        x3i = a[idx1 + 1] - a[idx3 + 1];
        a[offa] = x0r + x2r;
        a[offa + 1] = x0i + x2i;
        a[idx1] = x0r - x2r;
        a[idx1 + 1] = x0i - x2i;
        a[idx2] = x1r - x3i;
        a[idx2 + 1] = x1i + x3r;
        a[idx3] = x1r + x3i;
        a[idx3 + 1] = x1i - x3r;
        wn4r = w[startw + 1];
        k = 0;
        for (int j = 2; j < mh; j += 2) {
            k += 4;
            idx4 = startw + k;
            wk1r = w[idx4];
            wk1i = w[idx4 + 1];
            wk3r = w[idx4 + 2];
            wk3i = w[idx4 + 3];
            j1 = j + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            idx5 = offa + j;
            x0r = a[idx5] + a[idx2];
            x0i = a[idx5 + 1] + a[idx2 + 1];
            x1r = a[idx5] - a[idx2];
            x1i = a[idx5 + 1] - a[idx2 + 1];
            x2r = a[idx1] + a[idx3];
            x2i = a[idx1 + 1] + a[idx3 + 1];
            x3r = a[idx1] - a[idx3];
            x3i = a[idx1 + 1] - a[idx3 + 1];
            a[idx5] = x0r + x2r;
            a[idx5 + 1] = x0i + x2i;
            a[idx1] = x0r - x2r;
            a[idx1 + 1] = x0i - x2i;
            x0r = x1r - x3i;
            x0i = x1i + x3r;
            a[idx2] = wk1r * x0r - wk1i * x0i;
            a[idx2 + 1] = wk1r * x0i + wk1i * x0r;
            x0r = x1r + x3i;
            x0i = x1i - x3r;
            a[idx3] = wk3r * x0r + wk3i * x0i;
            a[idx3 + 1] = wk3r * x0i - wk3i * x0r;
            j0 = m - j;
            j1 = j0 + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx0 = offa + j0;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            x0r = a[idx0] + a[idx2];
            x0i = a[idx0 + 1] + a[idx2 + 1];
            x1r = a[idx0] - a[idx2];
            x1i = a[idx0 + 1] - a[idx2 + 1];
            x2r = a[idx1] + a[idx3];
            x2i = a[idx1 + 1] + a[idx3 + 1];
            x3r = a[idx1] - a[idx3];
            x3i = a[idx1 + 1] - a[idx3 + 1];
            a[idx0] = x0r + x2r;
            a[idx0 + 1] = x0i + x2i;
            a[idx1] = x0r - x2r;
            a[idx1 + 1] = x0i - x2i;
            x0r = x1r - x3i;
            x0i = x1i + x3r;
            a[idx2] = wk1i * x0r - wk1r * x0i;
            a[idx2 + 1] = wk1i * x0i + wk1r * x0r;
            x0r = x1r + x3i;
            x0i = x1i - x3r;
            a[idx3] = wk3i * x0r + wk3r * x0i;
            a[idx3 + 1] = wk3i * x0i - wk3r * x0r;
        }
        j0 = mh;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx0 = offa + j0;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a[idx0] + a[idx2];
        x0i = a[idx0 + 1] + a[idx2 + 1];
        x1r = a[idx0] - a[idx2];
        x1i = a[idx0 + 1] - a[idx2 + 1];
        x2r = a[idx1] + a[idx3];
        x2i = a[idx1 + 1] + a[idx3 + 1];
        x3r = a[idx1] - a[idx3];
        x3i = a[idx1 + 1] - a[idx3 + 1];
        a[idx0] = x0r + x2r;
        a[idx0 + 1] = x0i + x2i;
        a[idx1] = x0r - x2r;
        a[idx1 + 1] = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a[idx2] = wn4r * (x0r - x0i);
        a[idx2 + 1] = wn4r * (x0i + x0r);
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a[idx3] = -wn4r * (x0r + x0i);
        a[idx3 + 1] = -wn4r * (x0i - x0r);
    }

    public static void cftmdl1(long n, FloatLargeArray a, long offa, FloatLargeArray w, long startw)
    {
        long j0, j1, j2, j3, k, m, mh;
        float wn4r, wk1r, wk1i, wk3r, wk3i;
        float x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;
        long idx0, idx1, idx2, idx3, idx4, idx5;

        mh = n >> 3l;
        m = 2 * mh;
        j1 = m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a.getFloat(offa) + a.getFloat(idx2);
        x0i = a.getFloat(offa + 1) + a.getFloat(idx2 + 1);
        x1r = a.getFloat(offa) - a.getFloat(idx2);
        x1i = a.getFloat(offa + 1) - a.getFloat(idx2 + 1);
        x2r = a.getFloat(idx1) + a.getFloat(idx3);
        x2i = a.getFloat(idx1 + 1) + a.getFloat(idx3 + 1);
        x3r = a.getFloat(idx1) - a.getFloat(idx3);
        x3i = a.getFloat(idx1 + 1) - a.getFloat(idx3 + 1);
        a.setFloat(offa, x0r + x2r);
        a.setFloat(offa + 1, x0i + x2i);
        a.setFloat(idx1, x0r - x2r);
        a.setFloat(idx1 + 1, x0i - x2i);
        a.setFloat(idx2, x1r - x3i);
        a.setFloat(idx2 + 1, x1i + x3r);
        a.setFloat(idx3, x1r + x3i);
        a.setFloat(idx3 + 1, x1i - x3r);
        wn4r = w.getFloat(startw + 1);
        k = 0;
        for (long j = 2; j < mh; j += 2) {
            k += 4;
            idx4 = startw + k;
            wk1r = w.getFloat(idx4);
            wk1i = w.getFloat(idx4 + 1);
            wk3r = w.getFloat(idx4 + 2);
            wk3i = w.getFloat(idx4 + 3);
            j1 = j + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            idx5 = offa + j;
            x0r = a.getFloat(idx5) + a.getFloat(idx2);
            x0i = a.getFloat(idx5 + 1) + a.getFloat(idx2 + 1);
            x1r = a.getFloat(idx5) - a.getFloat(idx2);
            x1i = a.getFloat(idx5 + 1) - a.getFloat(idx2 + 1);
            x2r = a.getFloat(idx1) + a.getFloat(idx3);
            x2i = a.getFloat(idx1 + 1) + a.getFloat(idx3 + 1);
            x3r = a.getFloat(idx1) - a.getFloat(idx3);
            x3i = a.getFloat(idx1 + 1) - a.getFloat(idx3 + 1);
            a.setFloat(idx5, x0r + x2r);
            a.setFloat(idx5 + 1, x0i + x2i);
            a.setFloat(idx1, x0r - x2r);
            a.setFloat(idx1 + 1, x0i - x2i);
            x0r = x1r - x3i;
            x0i = x1i + x3r;
            a.setFloat(idx2, wk1r * x0r - wk1i * x0i);
            a.setFloat(idx2 + 1, wk1r * x0i + wk1i * x0r);
            x0r = x1r + x3i;
            x0i = x1i - x3r;
            a.setFloat(idx3, wk3r * x0r + wk3i * x0i);
            a.setFloat(idx3 + 1, wk3r * x0i - wk3i * x0r);
            j0 = m - j;
            j1 = j0 + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx0 = offa + j0;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            x0r = a.getFloat(idx0) + a.getFloat(idx2);
            x0i = a.getFloat(idx0 + 1) + a.getFloat(idx2 + 1);
            x1r = a.getFloat(idx0) - a.getFloat(idx2);
            x1i = a.getFloat(idx0 + 1) - a.getFloat(idx2 + 1);
            x2r = a.getFloat(idx1) + a.getFloat(idx3);
            x2i = a.getFloat(idx1 + 1) + a.getFloat(idx3 + 1);
            x3r = a.getFloat(idx1) - a.getFloat(idx3);
            x3i = a.getFloat(idx1 + 1) - a.getFloat(idx3 + 1);
            a.setFloat(idx0, x0r + x2r);
            a.setFloat(idx0 + 1, x0i + x2i);
            a.setFloat(idx1, x0r - x2r);
            a.setFloat(idx1 + 1, x0i - x2i);
            x0r = x1r - x3i;
            x0i = x1i + x3r;
            a.setFloat(idx2, wk1i * x0r - wk1r * x0i);
            a.setFloat(idx2 + 1, wk1i * x0i + wk1r * x0r);
            x0r = x1r + x3i;
            x0i = x1i - x3r;
            a.setFloat(idx3, wk3i * x0r + wk3r * x0i);
            a.setFloat(idx3 + 1, wk3i * x0i - wk3r * x0r);
        }
        j0 = mh;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx0 = offa + j0;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a.getFloat(idx0) + a.getFloat(idx2);
        x0i = a.getFloat(idx0 + 1) + a.getFloat(idx2 + 1);
        x1r = a.getFloat(idx0) - a.getFloat(idx2);
        x1i = a.getFloat(idx0 + 1) - a.getFloat(idx2 + 1);
        x2r = a.getFloat(idx1) + a.getFloat(idx3);
        x2i = a.getFloat(idx1 + 1) + a.getFloat(idx3 + 1);
        x3r = a.getFloat(idx1) - a.getFloat(idx3);
        x3i = a.getFloat(idx1 + 1) - a.getFloat(idx3 + 1);
        a.setFloat(idx0, x0r + x2r);
        a.setFloat(idx0 + 1, x0i + x2i);
        a.setFloat(idx1, x0r - x2r);
        a.setFloat(idx1 + 1, x0i - x2i);
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a.setFloat(idx2, wn4r * (x0r - x0i));
        a.setFloat(idx2 + 1, wn4r * (x0i + x0r));
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a.setFloat(idx3, -wn4r * (x0r + x0i));
        a.setFloat(idx3 + 1, -wn4r * (x0i - x0r));
    }

    public static void cftmdl2(int n, float[] a, int offa, float[] w, int startw)
    {
        int j0, j1, j2, j3, k, kr, m, mh;
        float wn4r, wk1r, wk1i, wk3r, wk3i, wd1r, wd1i, wd3r, wd3i;
        float x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, y0r, y0i, y2r, y2i;
        int idx0, idx1, idx2, idx3, idx4, idx5, idx6;

        mh = n >> 3;
        m = 2 * mh;
        wn4r = w[startw + 1];
        j1 = m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a[offa] - a[idx2 + 1];
        x0i = a[offa + 1] + a[idx2];
        x1r = a[offa] + a[idx2 + 1];
        x1i = a[offa + 1] - a[idx2];
        x2r = a[idx1] - a[idx3 + 1];
        x2i = a[idx1 + 1] + a[idx3];
        x3r = a[idx1] + a[idx3 + 1];
        x3i = a[idx1 + 1] - a[idx3];
        y0r = wn4r * (x2r - x2i);
        y0i = wn4r * (x2i + x2r);
        a[offa] = x0r + y0r;
        a[offa + 1] = x0i + y0i;
        a[idx1] = x0r - y0r;
        a[idx1 + 1] = x0i - y0i;
        y0r = wn4r * (x3r - x3i);
        y0i = wn4r * (x3i + x3r);
        a[idx2] = x1r - y0i;
        a[idx2 + 1] = x1i + y0r;
        a[idx3] = x1r + y0i;
        a[idx3 + 1] = x1i - y0r;
        k = 0;
        kr = 2 * m;
        for (int j = 2; j < mh; j += 2) {
            k += 4;
            idx4 = startw + k;
            wk1r = w[idx4];
            wk1i = w[idx4 + 1];
            wk3r = w[idx4 + 2];
            wk3i = w[idx4 + 3];
            kr -= 4;
            idx5 = startw + kr;
            wd1i = w[idx5];
            wd1r = w[idx5 + 1];
            wd3i = w[idx5 + 2];
            wd3r = w[idx5 + 3];
            j1 = j + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            idx6 = offa + j;
            x0r = a[idx6] - a[idx2 + 1];
            x0i = a[idx6 + 1] + a[idx2];
            x1r = a[idx6] + a[idx2 + 1];
            x1i = a[idx6 + 1] - a[idx2];
            x2r = a[idx1] - a[idx3 + 1];
            x2i = a[idx1 + 1] + a[idx3];
            x3r = a[idx1] + a[idx3 + 1];
            x3i = a[idx1 + 1] - a[idx3];
            y0r = wk1r * x0r - wk1i * x0i;
            y0i = wk1r * x0i + wk1i * x0r;
            y2r = wd1r * x2r - wd1i * x2i;
            y2i = wd1r * x2i + wd1i * x2r;
            a[idx6] = y0r + y2r;
            a[idx6 + 1] = y0i + y2i;
            a[idx1] = y0r - y2r;
            a[idx1 + 1] = y0i - y2i;
            y0r = wk3r * x1r + wk3i * x1i;
            y0i = wk3r * x1i - wk3i * x1r;
            y2r = wd3r * x3r + wd3i * x3i;
            y2i = wd3r * x3i - wd3i * x3r;
            a[idx2] = y0r + y2r;
            a[idx2 + 1] = y0i + y2i;
            a[idx3] = y0r - y2r;
            a[idx3 + 1] = y0i - y2i;
            j0 = m - j;
            j1 = j0 + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx0 = offa + j0;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            x0r = a[idx0] - a[idx2 + 1];
            x0i = a[idx0 + 1] + a[idx2];
            x1r = a[idx0] + a[idx2 + 1];
            x1i = a[idx0 + 1] - a[idx2];
            x2r = a[idx1] - a[idx3 + 1];
            x2i = a[idx1 + 1] + a[idx3];
            x3r = a[idx1] + a[idx3 + 1];
            x3i = a[idx1 + 1] - a[idx3];
            y0r = wd1i * x0r - wd1r * x0i;
            y0i = wd1i * x0i + wd1r * x0r;
            y2r = wk1i * x2r - wk1r * x2i;
            y2i = wk1i * x2i + wk1r * x2r;
            a[idx0] = y0r + y2r;
            a[idx0 + 1] = y0i + y2i;
            a[idx1] = y0r - y2r;
            a[idx1 + 1] = y0i - y2i;
            y0r = wd3i * x1r + wd3r * x1i;
            y0i = wd3i * x1i - wd3r * x1r;
            y2r = wk3i * x3r + wk3r * x3i;
            y2i = wk3i * x3i - wk3r * x3r;
            a[idx2] = y0r + y2r;
            a[idx2 + 1] = y0i + y2i;
            a[idx3] = y0r - y2r;
            a[idx3 + 1] = y0i - y2i;
        }
        wk1r = w[startw + m];
        wk1i = w[startw + m + 1];
        j0 = mh;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx0 = offa + j0;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a[idx0] - a[idx2 + 1];
        x0i = a[idx0 + 1] + a[idx2];
        x1r = a[idx0] + a[idx2 + 1];
        x1i = a[idx0 + 1] - a[idx2];
        x2r = a[idx1] - a[idx3 + 1];
        x2i = a[idx1 + 1] + a[idx3];
        x3r = a[idx1] + a[idx3 + 1];
        x3i = a[idx1 + 1] - a[idx3];
        y0r = wk1r * x0r - wk1i * x0i;
        y0i = wk1r * x0i + wk1i * x0r;
        y2r = wk1i * x2r - wk1r * x2i;
        y2i = wk1i * x2i + wk1r * x2r;
        a[idx0] = y0r + y2r;
        a[idx0 + 1] = y0i + y2i;
        a[idx1] = y0r - y2r;
        a[idx1 + 1] = y0i - y2i;
        y0r = wk1i * x1r - wk1r * x1i;
        y0i = wk1i * x1i + wk1r * x1r;
        y2r = wk1r * x3r - wk1i * x3i;
        y2i = wk1r * x3i + wk1i * x3r;
        a[idx2] = y0r - y2r;
        a[idx2 + 1] = y0i - y2i;
        a[idx3] = y0r + y2r;
        a[idx3 + 1] = y0i + y2i;
    }

    public static void cftmdl2(long n, FloatLargeArray a, long offa, FloatLargeArray w, long startw)
    {
        long j0, j1, j2, j3, k, kr, m, mh;
        float wn4r, wk1r, wk1i, wk3r, wk3i, wd1r, wd1i, wd3r, wd3i;
        float x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, y0r, y0i, y2r, y2i;
        long idx0, idx1, idx2, idx3, idx4, idx5, idx6;

        mh = n >> 3l;
        m = 2 * mh;
        wn4r = w.getFloat(startw + 1);
        j1 = m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a.getFloat(offa) - a.getFloat(idx2 + 1);
        x0i = a.getFloat(offa + 1) + a.getFloat(idx2);
        x1r = a.getFloat(offa) + a.getFloat(idx2 + 1);
        x1i = a.getFloat(offa + 1) - a.getFloat(idx2);
        x2r = a.getFloat(idx1) - a.getFloat(idx3 + 1);
        x2i = a.getFloat(idx1 + 1) + a.getFloat(idx3);
        x3r = a.getFloat(idx1) + a.getFloat(idx3 + 1);
        x3i = a.getFloat(idx1 + 1) - a.getFloat(idx3);
        y0r = wn4r * (x2r - x2i);
        y0i = wn4r * (x2i + x2r);
        a.setFloat(offa, x0r + y0r);
        a.setFloat(offa + 1, x0i + y0i);
        a.setFloat(idx1, x0r - y0r);
        a.setFloat(idx1 + 1, x0i - y0i);
        y0r = wn4r * (x3r - x3i);
        y0i = wn4r * (x3i + x3r);
        a.setFloat(idx2, x1r - y0i);
        a.setFloat(idx2 + 1, x1i + y0r);
        a.setFloat(idx3, x1r + y0i);
        a.setFloat(idx3 + 1, x1i - y0r);
        k = 0;
        kr = 2 * m;
        for (int j = 2; j < mh; j += 2) {
            k += 4;
            idx4 = startw + k;
            wk1r = w.getFloat(idx4);
            wk1i = w.getFloat(idx4 + 1);
            wk3r = w.getFloat(idx4 + 2);
            wk3i = w.getFloat(idx4 + 3);
            kr -= 4;
            idx5 = startw + kr;
            wd1i = w.getFloat(idx5);
            wd1r = w.getFloat(idx5 + 1);
            wd3i = w.getFloat(idx5 + 2);
            wd3r = w.getFloat(idx5 + 3);
            j1 = j + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            idx6 = offa + j;
            x0r = a.getFloat(idx6) - a.getFloat(idx2 + 1);
            x0i = a.getFloat(idx6 + 1) + a.getFloat(idx2);
            x1r = a.getFloat(idx6) + a.getFloat(idx2 + 1);
            x1i = a.getFloat(idx6 + 1) - a.getFloat(idx2);
            x2r = a.getFloat(idx1) - a.getFloat(idx3 + 1);
            x2i = a.getFloat(idx1 + 1) + a.getFloat(idx3);
            x3r = a.getFloat(idx1) + a.getFloat(idx3 + 1);
            x3i = a.getFloat(idx1 + 1) - a.getFloat(idx3);
            y0r = wk1r * x0r - wk1i * x0i;
            y0i = wk1r * x0i + wk1i * x0r;
            y2r = wd1r * x2r - wd1i * x2i;
            y2i = wd1r * x2i + wd1i * x2r;
            a.setFloat(idx6, y0r + y2r);
            a.setFloat(idx6 + 1, y0i + y2i);
            a.setFloat(idx1, y0r - y2r);
            a.setFloat(idx1 + 1, y0i - y2i);
            y0r = wk3r * x1r + wk3i * x1i;
            y0i = wk3r * x1i - wk3i * x1r;
            y2r = wd3r * x3r + wd3i * x3i;
            y2i = wd3r * x3i - wd3i * x3r;
            a.setFloat(idx2, y0r + y2r);
            a.setFloat(idx2 + 1, y0i + y2i);
            a.setFloat(idx3, y0r - y2r);
            a.setFloat(idx3 + 1, y0i - y2i);
            j0 = m - j;
            j1 = j0 + m;
            j2 = j1 + m;
            j3 = j2 + m;
            idx0 = offa + j0;
            idx1 = offa + j1;
            idx2 = offa + j2;
            idx3 = offa + j3;
            x0r = a.getFloat(idx0) - a.getFloat(idx2 + 1);
            x0i = a.getFloat(idx0 + 1) + a.getFloat(idx2);
            x1r = a.getFloat(idx0) + a.getFloat(idx2 + 1);
            x1i = a.getFloat(idx0 + 1) - a.getFloat(idx2);
            x2r = a.getFloat(idx1) - a.getFloat(idx3 + 1);
            x2i = a.getFloat(idx1 + 1) + a.getFloat(idx3);
            x3r = a.getFloat(idx1) + a.getFloat(idx3 + 1);
            x3i = a.getFloat(idx1 + 1) - a.getFloat(idx3);
            y0r = wd1i * x0r - wd1r * x0i;
            y0i = wd1i * x0i + wd1r * x0r;
            y2r = wk1i * x2r - wk1r * x2i;
            y2i = wk1i * x2i + wk1r * x2r;
            a.setFloat(idx0, y0r + y2r);
            a.setFloat(idx0 + 1, y0i + y2i);
            a.setFloat(idx1, y0r - y2r);
            a.setFloat(idx1 + 1, y0i - y2i);
            y0r = wd3i * x1r + wd3r * x1i;
            y0i = wd3i * x1i - wd3r * x1r;
            y2r = wk3i * x3r + wk3r * x3i;
            y2i = wk3i * x3i - wk3r * x3r;
            a.setFloat(idx2, y0r + y2r);
            a.setFloat(idx2 + 1, y0i + y2i);
            a.setFloat(idx3, y0r - y2r);
            a.setFloat(idx3 + 1, y0i - y2i);
        }
        wk1r = w.getFloat(startw + m);
        wk1i = w.getFloat(startw + m + 1);
        j0 = mh;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        idx0 = offa + j0;
        idx1 = offa + j1;
        idx2 = offa + j2;
        idx3 = offa + j3;
        x0r = a.getFloat(idx0) - a.getFloat(idx2 + 1);
        x0i = a.getFloat(idx0 + 1) + a.getFloat(idx2);
        x1r = a.getFloat(idx0) + a.getFloat(idx2 + 1);
        x1i = a.getFloat(idx0 + 1) - a.getFloat(idx2);
        x2r = a.getFloat(idx1) - a.getFloat(idx3 + 1);
        x2i = a.getFloat(idx1 + 1) + a.getFloat(idx3);
        x3r = a.getFloat(idx1) + a.getFloat(idx3 + 1);
        x3i = a.getFloat(idx1 + 1) - a.getFloat(idx3);
        y0r = wk1r * x0r - wk1i * x0i;
        y0i = wk1r * x0i + wk1i * x0r;
        y2r = wk1i * x2r - wk1r * x2i;
        y2i = wk1i * x2i + wk1r * x2r;
        a.setFloat(idx0, y0r + y2r);
        a.setFloat(idx0 + 1, y0i + y2i);
        a.setFloat(idx1, y0r - y2r);
        a.setFloat(idx1 + 1, y0i - y2i);
        y0r = wk1i * x1r - wk1r * x1i;
        y0i = wk1i * x1i + wk1r * x1r;
        y2r = wk1r * x3r - wk1i * x3i;
        y2i = wk1r * x3i + wk1i * x3r;
        a.setFloat(idx2, y0r - y2r);
        a.setFloat(idx2 + 1, y0i - y2i);
        a.setFloat(idx3, y0r + y2r);
        a.setFloat(idx3 + 1, y0i + y2i);
    }

    public static void cftfx41(int n, float[] a, int offa, int nw, float[] w)
    {
        if (n == 128) {
            cftf161(a, offa, w, nw - 8);
            cftf162(a, offa + 32, w, nw - 32);
            cftf161(a, offa + 64, w, nw - 8);
            cftf161(a, offa + 96, w, nw - 8);
        } else {
            cftf081(a, offa, w, nw - 8);
            cftf082(a, offa + 16, w, nw - 8);
            cftf081(a, offa + 32, w, nw - 8);
            cftf081(a, offa + 48, w, nw - 8);
        }
    }

    public static void cftfx41(long n, FloatLargeArray a, long offa, long nw, FloatLargeArray w)
    {
        if (n == 128) {
            cftf161(a, offa, w, nw - 8);
            cftf162(a, offa + 32, w, nw - 32);
            cftf161(a, offa + 64, w, nw - 8);
            cftf161(a, offa + 96, w, nw - 8);
        } else {
            cftf081(a, offa, w, nw - 8);
            cftf082(a, offa + 16, w, nw - 8);
            cftf081(a, offa + 32, w, nw - 8);
            cftf081(a, offa + 48, w, nw - 8);
        }
    }

    public static void cftf161(float[] a, int offa, float[] w, int startw)
    {
        float wn4r, wk1r, wk1i, x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i, y8r, y8i, y9r, y9i, y10r, y10i, y11r, y11i, y12r, y12i, y13r, y13i, y14r, y14i, y15r, y15i;

        wn4r = w[startw + 1];
        wk1r = w[startw + 2];
        wk1i = w[startw + 3];

        x0r = a[offa] + a[offa + 16];
        x0i = a[offa + 1] + a[offa + 17];
        x1r = a[offa] - a[offa + 16];
        x1i = a[offa + 1] - a[offa + 17];
        x2r = a[offa + 8] + a[offa + 24];
        x2i = a[offa + 9] + a[offa + 25];
        x3r = a[offa + 8] - a[offa + 24];
        x3i = a[offa + 9] - a[offa + 25];
        y0r = x0r + x2r;
        y0i = x0i + x2i;
        y4r = x0r - x2r;
        y4i = x0i - x2i;
        y8r = x1r - x3i;
        y8i = x1i + x3r;
        y12r = x1r + x3i;
        y12i = x1i - x3r;
        x0r = a[offa + 2] + a[offa + 18];
        x0i = a[offa + 3] + a[offa + 19];
        x1r = a[offa + 2] - a[offa + 18];
        x1i = a[offa + 3] - a[offa + 19];
        x2r = a[offa + 10] + a[offa + 26];
        x2i = a[offa + 11] + a[offa + 27];
        x3r = a[offa + 10] - a[offa + 26];
        x3i = a[offa + 11] - a[offa + 27];
        y1r = x0r + x2r;
        y1i = x0i + x2i;
        y5r = x0r - x2r;
        y5i = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        y9r = wk1r * x0r - wk1i * x0i;
        y9i = wk1r * x0i + wk1i * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        y13r = wk1i * x0r - wk1r * x0i;
        y13i = wk1i * x0i + wk1r * x0r;
        x0r = a[offa + 4] + a[offa + 20];
        x0i = a[offa + 5] + a[offa + 21];
        x1r = a[offa + 4] - a[offa + 20];
        x1i = a[offa + 5] - a[offa + 21];
        x2r = a[offa + 12] + a[offa + 28];
        x2i = a[offa + 13] + a[offa + 29];
        x3r = a[offa + 12] - a[offa + 28];
        x3i = a[offa + 13] - a[offa + 29];
        y2r = x0r + x2r;
        y2i = x0i + x2i;
        y6r = x0r - x2r;
        y6i = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        y10r = wn4r * (x0r - x0i);
        y10i = wn4r * (x0i + x0r);
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        y14r = wn4r * (x0r + x0i);
        y14i = wn4r * (x0i - x0r);
        x0r = a[offa + 6] + a[offa + 22];
        x0i = a[offa + 7] + a[offa + 23];
        x1r = a[offa + 6] - a[offa + 22];
        x1i = a[offa + 7] - a[offa + 23];
        x2r = a[offa + 14] + a[offa + 30];
        x2i = a[offa + 15] + a[offa + 31];
        x3r = a[offa + 14] - a[offa + 30];
        x3i = a[offa + 15] - a[offa + 31];
        y3r = x0r + x2r;
        y3i = x0i + x2i;
        y7r = x0r - x2r;
        y7i = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        y11r = wk1i * x0r - wk1r * x0i;
        y11i = wk1i * x0i + wk1r * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        y15r = wk1r * x0r - wk1i * x0i;
        y15i = wk1r * x0i + wk1i * x0r;
        x0r = y12r - y14r;
        x0i = y12i - y14i;
        x1r = y12r + y14r;
        x1i = y12i + y14i;
        x2r = y13r - y15r;
        x2i = y13i - y15i;
        x3r = y13r + y15r;
        x3i = y13i + y15i;
        a[offa + 24] = x0r + x2r;
        a[offa + 25] = x0i + x2i;
        a[offa + 26] = x0r - x2r;
        a[offa + 27] = x0i - x2i;
        a[offa + 28] = x1r - x3i;
        a[offa + 29] = x1i + x3r;
        a[offa + 30] = x1r + x3i;
        a[offa + 31] = x1i - x3r;
        x0r = y8r + y10r;
        x0i = y8i + y10i;
        x1r = y8r - y10r;
        x1i = y8i - y10i;
        x2r = y9r + y11r;
        x2i = y9i + y11i;
        x3r = y9r - y11r;
        x3i = y9i - y11i;
        a[offa + 16] = x0r + x2r;
        a[offa + 17] = x0i + x2i;
        a[offa + 18] = x0r - x2r;
        a[offa + 19] = x0i - x2i;
        a[offa + 20] = x1r - x3i;
        a[offa + 21] = x1i + x3r;
        a[offa + 22] = x1r + x3i;
        a[offa + 23] = x1i - x3r;
        x0r = y5r - y7i;
        x0i = y5i + y7r;
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        x0r = y5r + y7i;
        x0i = y5i - y7r;
        x3r = wn4r * (x0r - x0i);
        x3i = wn4r * (x0i + x0r);
        x0r = y4r - y6i;
        x0i = y4i + y6r;
        x1r = y4r + y6i;
        x1i = y4i - y6r;
        a[offa + 8] = x0r + x2r;
        a[offa + 9] = x0i + x2i;
        a[offa + 10] = x0r - x2r;
        a[offa + 11] = x0i - x2i;
        a[offa + 12] = x1r - x3i;
        a[offa + 13] = x1i + x3r;
        a[offa + 14] = x1r + x3i;
        a[offa + 15] = x1i - x3r;
        x0r = y0r + y2r;
        x0i = y0i + y2i;
        x1r = y0r - y2r;
        x1i = y0i - y2i;
        x2r = y1r + y3r;
        x2i = y1i + y3i;
        x3r = y1r - y3r;
        x3i = y1i - y3i;
        a[offa] = x0r + x2r;
        a[offa + 1] = x0i + x2i;
        a[offa + 2] = x0r - x2r;
        a[offa + 3] = x0i - x2i;
        a[offa + 4] = x1r - x3i;
        a[offa + 5] = x1i + x3r;
        a[offa + 6] = x1r + x3i;
        a[offa + 7] = x1i - x3r;
    }

    public static void cftf161(FloatLargeArray a, long offa, FloatLargeArray w, long startw)
    {
        float wn4r, wk1r, wk1i, x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i, y8r, y8i, y9r, y9i, y10r, y10i, y11r, y11i, y12r, y12i, y13r, y13i, y14r, y14i, y15r, y15i;

        wn4r = w.getFloat(startw + 1);
        wk1r = w.getFloat(startw + 2);
        wk1i = w.getFloat(startw + 3);

        x0r = a.getFloat(offa) + a.getFloat(offa + 16);
        x0i = a.getFloat(offa + 1) + a.getFloat(offa + 17);
        x1r = a.getFloat(offa) - a.getFloat(offa + 16);
        x1i = a.getFloat(offa + 1) - a.getFloat(offa + 17);
        x2r = a.getFloat(offa + 8) + a.getFloat(offa + 24);
        x2i = a.getFloat(offa + 9) + a.getFloat(offa + 25);
        x3r = a.getFloat(offa + 8) - a.getFloat(offa + 24);
        x3i = a.getFloat(offa + 9) - a.getFloat(offa + 25);
        y0r = x0r + x2r;
        y0i = x0i + x2i;
        y4r = x0r - x2r;
        y4i = x0i - x2i;
        y8r = x1r - x3i;
        y8i = x1i + x3r;
        y12r = x1r + x3i;
        y12i = x1i - x3r;
        x0r = a.getFloat(offa + 2) + a.getFloat(offa + 18);
        x0i = a.getFloat(offa + 3) + a.getFloat(offa + 19);
        x1r = a.getFloat(offa + 2) - a.getFloat(offa + 18);
        x1i = a.getFloat(offa + 3) - a.getFloat(offa + 19);
        x2r = a.getFloat(offa + 10) + a.getFloat(offa + 26);
        x2i = a.getFloat(offa + 11) + a.getFloat(offa + 27);
        x3r = a.getFloat(offa + 10) - a.getFloat(offa + 26);
        x3i = a.getFloat(offa + 11) - a.getFloat(offa + 27);
        y1r = x0r + x2r;
        y1i = x0i + x2i;
        y5r = x0r - x2r;
        y5i = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        y9r = wk1r * x0r - wk1i * x0i;
        y9i = wk1r * x0i + wk1i * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        y13r = wk1i * x0r - wk1r * x0i;
        y13i = wk1i * x0i + wk1r * x0r;
        x0r = a.getFloat(offa + 4) + a.getFloat(offa + 20);
        x0i = a.getFloat(offa + 5) + a.getFloat(offa + 21);
        x1r = a.getFloat(offa + 4) - a.getFloat(offa + 20);
        x1i = a.getFloat(offa + 5) - a.getFloat(offa + 21);
        x2r = a.getFloat(offa + 12) + a.getFloat(offa + 28);
        x2i = a.getFloat(offa + 13) + a.getFloat(offa + 29);
        x3r = a.getFloat(offa + 12) - a.getFloat(offa + 28);
        x3i = a.getFloat(offa + 13) - a.getFloat(offa + 29);
        y2r = x0r + x2r;
        y2i = x0i + x2i;
        y6r = x0r - x2r;
        y6i = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        y10r = wn4r * (x0r - x0i);
        y10i = wn4r * (x0i + x0r);
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        y14r = wn4r * (x0r + x0i);
        y14i = wn4r * (x0i - x0r);
        x0r = a.getFloat(offa + 6) + a.getFloat(offa + 22);
        x0i = a.getFloat(offa + 7) + a.getFloat(offa + 23);
        x1r = a.getFloat(offa + 6) - a.getFloat(offa + 22);
        x1i = a.getFloat(offa + 7) - a.getFloat(offa + 23);
        x2r = a.getFloat(offa + 14) + a.getFloat(offa + 30);
        x2i = a.getFloat(offa + 15) + a.getFloat(offa + 31);
        x3r = a.getFloat(offa + 14) - a.getFloat(offa + 30);
        x3i = a.getFloat(offa + 15) - a.getFloat(offa + 31);
        y3r = x0r + x2r;
        y3i = x0i + x2i;
        y7r = x0r - x2r;
        y7i = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        y11r = wk1i * x0r - wk1r * x0i;
        y11i = wk1i * x0i + wk1r * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        y15r = wk1r * x0r - wk1i * x0i;
        y15i = wk1r * x0i + wk1i * x0r;
        x0r = y12r - y14r;
        x0i = y12i - y14i;
        x1r = y12r + y14r;
        x1i = y12i + y14i;
        x2r = y13r - y15r;
        x2i = y13i - y15i;
        x3r = y13r + y15r;
        x3i = y13i + y15i;
        a.setFloat(offa + 24, x0r + x2r);
        a.setFloat(offa + 25, x0i + x2i);
        a.setFloat(offa + 26, x0r - x2r);
        a.setFloat(offa + 27, x0i - x2i);
        a.setFloat(offa + 28, x1r - x3i);
        a.setFloat(offa + 29, x1i + x3r);
        a.setFloat(offa + 30, x1r + x3i);
        a.setFloat(offa + 31, x1i - x3r);
        x0r = y8r + y10r;
        x0i = y8i + y10i;
        x1r = y8r - y10r;
        x1i = y8i - y10i;
        x2r = y9r + y11r;
        x2i = y9i + y11i;
        x3r = y9r - y11r;
        x3i = y9i - y11i;
        a.setFloat(offa + 16, x0r + x2r);
        a.setFloat(offa + 17, x0i + x2i);
        a.setFloat(offa + 18, x0r - x2r);
        a.setFloat(offa + 19, x0i - x2i);
        a.setFloat(offa + 20, x1r - x3i);
        a.setFloat(offa + 21, x1i + x3r);
        a.setFloat(offa + 22, x1r + x3i);
        a.setFloat(offa + 23, x1i - x3r);
        x0r = y5r - y7i;
        x0i = y5i + y7r;
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        x0r = y5r + y7i;
        x0i = y5i - y7r;
        x3r = wn4r * (x0r - x0i);
        x3i = wn4r * (x0i + x0r);
        x0r = y4r - y6i;
        x0i = y4i + y6r;
        x1r = y4r + y6i;
        x1i = y4i - y6r;
        a.setFloat(offa + 8, x0r + x2r);
        a.setFloat(offa + 9, x0i + x2i);
        a.setFloat(offa + 10, x0r - x2r);
        a.setFloat(offa + 11, x0i - x2i);
        a.setFloat(offa + 12, x1r - x3i);
        a.setFloat(offa + 13, x1i + x3r);
        a.setFloat(offa + 14, x1r + x3i);
        a.setFloat(offa + 15, x1i - x3r);
        x0r = y0r + y2r;
        x0i = y0i + y2i;
        x1r = y0r - y2r;
        x1i = y0i - y2i;
        x2r = y1r + y3r;
        x2i = y1i + y3i;
        x3r = y1r - y3r;
        x3i = y1i - y3i;
        a.setFloat(offa, x0r + x2r);
        a.setFloat(offa + 1, x0i + x2i);
        a.setFloat(offa + 2, x0r - x2r);
        a.setFloat(offa + 3, x0i - x2i);
        a.setFloat(offa + 4, x1r - x3i);
        a.setFloat(offa + 5, x1i + x3r);
        a.setFloat(offa + 6, x1r + x3i);
        a.setFloat(offa + 7, x1i - x3r);
    }

    public static void cftf162(float[] a, int offa, float[] w, int startw)
    {
        float wn4r, wk1r, wk1i, wk2r, wk2i, wk3r, wk3i, x0r, x0i, x1r, x1i, x2r, x2i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i, y8r, y8i, y9r, y9i, y10r, y10i, y11r, y11i, y12r, y12i, y13r, y13i, y14r, y14i, y15r, y15i;

        wn4r = w[startw + 1];
        wk1r = w[startw + 4];
        wk1i = w[startw + 5];
        wk3r = w[startw + 6];
        wk3i = -w[startw + 7];
        wk2r = w[startw + 8];
        wk2i = w[startw + 9];
        x1r = a[offa] - a[offa + 17];
        x1i = a[offa + 1] + a[offa + 16];
        x0r = a[offa + 8] - a[offa + 25];
        x0i = a[offa + 9] + a[offa + 24];
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        y0r = x1r + x2r;
        y0i = x1i + x2i;
        y4r = x1r - x2r;
        y4i = x1i - x2i;
        x1r = a[offa] + a[offa + 17];
        x1i = a[offa + 1] - a[offa + 16];
        x0r = a[offa + 8] + a[offa + 25];
        x0i = a[offa + 9] - a[offa + 24];
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        y8r = x1r - x2i;
        y8i = x1i + x2r;
        y12r = x1r + x2i;
        y12i = x1i - x2r;
        x0r = a[offa + 2] - a[offa + 19];
        x0i = a[offa + 3] + a[offa + 18];
        x1r = wk1r * x0r - wk1i * x0i;
        x1i = wk1r * x0i + wk1i * x0r;
        x0r = a[offa + 10] - a[offa + 27];
        x0i = a[offa + 11] + a[offa + 26];
        x2r = wk3i * x0r - wk3r * x0i;
        x2i = wk3i * x0i + wk3r * x0r;
        y1r = x1r + x2r;
        y1i = x1i + x2i;
        y5r = x1r - x2r;
        y5i = x1i - x2i;
        x0r = a[offa + 2] + a[offa + 19];
        x0i = a[offa + 3] - a[offa + 18];
        x1r = wk3r * x0r - wk3i * x0i;
        x1i = wk3r * x0i + wk3i * x0r;
        x0r = a[offa + 10] + a[offa + 27];
        x0i = a[offa + 11] - a[offa + 26];
        x2r = wk1r * x0r + wk1i * x0i;
        x2i = wk1r * x0i - wk1i * x0r;
        y9r = x1r - x2r;
        y9i = x1i - x2i;
        y13r = x1r + x2r;
        y13i = x1i + x2i;
        x0r = a[offa + 4] - a[offa + 21];
        x0i = a[offa + 5] + a[offa + 20];
        x1r = wk2r * x0r - wk2i * x0i;
        x1i = wk2r * x0i + wk2i * x0r;
        x0r = a[offa + 12] - a[offa + 29];
        x0i = a[offa + 13] + a[offa + 28];
        x2r = wk2i * x0r - wk2r * x0i;
        x2i = wk2i * x0i + wk2r * x0r;
        y2r = x1r + x2r;
        y2i = x1i + x2i;
        y6r = x1r - x2r;
        y6i = x1i - x2i;
        x0r = a[offa + 4] + a[offa + 21];
        x0i = a[offa + 5] - a[offa + 20];
        x1r = wk2i * x0r - wk2r * x0i;
        x1i = wk2i * x0i + wk2r * x0r;
        x0r = a[offa + 12] + a[offa + 29];
        x0i = a[offa + 13] - a[offa + 28];
        x2r = wk2r * x0r - wk2i * x0i;
        x2i = wk2r * x0i + wk2i * x0r;
        y10r = x1r - x2r;
        y10i = x1i - x2i;
        y14r = x1r + x2r;
        y14i = x1i + x2i;
        x0r = a[offa + 6] - a[offa + 23];
        x0i = a[offa + 7] + a[offa + 22];
        x1r = wk3r * x0r - wk3i * x0i;
        x1i = wk3r * x0i + wk3i * x0r;
        x0r = a[offa + 14] - a[offa + 31];
        x0i = a[offa + 15] + a[offa + 30];
        x2r = wk1i * x0r - wk1r * x0i;
        x2i = wk1i * x0i + wk1r * x0r;
        y3r = x1r + x2r;
        y3i = x1i + x2i;
        y7r = x1r - x2r;
        y7i = x1i - x2i;
        x0r = a[offa + 6] + a[offa + 23];
        x0i = a[offa + 7] - a[offa + 22];
        x1r = wk1i * x0r + wk1r * x0i;
        x1i = wk1i * x0i - wk1r * x0r;
        x0r = a[offa + 14] + a[offa + 31];
        x0i = a[offa + 15] - a[offa + 30];
        x2r = wk3i * x0r - wk3r * x0i;
        x2i = wk3i * x0i + wk3r * x0r;
        y11r = x1r + x2r;
        y11i = x1i + x2i;
        y15r = x1r - x2r;
        y15i = x1i - x2i;
        x1r = y0r + y2r;
        x1i = y0i + y2i;
        x2r = y1r + y3r;
        x2i = y1i + y3i;
        a[offa] = x1r + x2r;
        a[offa + 1] = x1i + x2i;
        a[offa + 2] = x1r - x2r;
        a[offa + 3] = x1i - x2i;
        x1r = y0r - y2r;
        x1i = y0i - y2i;
        x2r = y1r - y3r;
        x2i = y1i - y3i;
        a[offa + 4] = x1r - x2i;
        a[offa + 5] = x1i + x2r;
        a[offa + 6] = x1r + x2i;
        a[offa + 7] = x1i - x2r;
        x1r = y4r - y6i;
        x1i = y4i + y6r;
        x0r = y5r - y7i;
        x0i = y5i + y7r;
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        a[offa + 8] = x1r + x2r;
        a[offa + 9] = x1i + x2i;
        a[offa + 10] = x1r - x2r;
        a[offa + 11] = x1i - x2i;
        x1r = y4r + y6i;
        x1i = y4i - y6r;
        x0r = y5r + y7i;
        x0i = y5i - y7r;
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        a[offa + 12] = x1r - x2i;
        a[offa + 13] = x1i + x2r;
        a[offa + 14] = x1r + x2i;
        a[offa + 15] = x1i - x2r;
        x1r = y8r + y10r;
        x1i = y8i + y10i;
        x2r = y9r - y11r;
        x2i = y9i - y11i;
        a[offa + 16] = x1r + x2r;
        a[offa + 17] = x1i + x2i;
        a[offa + 18] = x1r - x2r;
        a[offa + 19] = x1i - x2i;
        x1r = y8r - y10r;
        x1i = y8i - y10i;
        x2r = y9r + y11r;
        x2i = y9i + y11i;
        a[offa + 20] = x1r - x2i;
        a[offa + 21] = x1i + x2r;
        a[offa + 22] = x1r + x2i;
        a[offa + 23] = x1i - x2r;
        x1r = y12r - y14i;
        x1i = y12i + y14r;
        x0r = y13r + y15i;
        x0i = y13i - y15r;
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        a[offa + 24] = x1r + x2r;
        a[offa + 25] = x1i + x2i;
        a[offa + 26] = x1r - x2r;
        a[offa + 27] = x1i - x2i;
        x1r = y12r + y14i;
        x1i = y12i - y14r;
        x0r = y13r - y15i;
        x0i = y13i + y15r;
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        a[offa + 28] = x1r - x2i;
        a[offa + 29] = x1i + x2r;
        a[offa + 30] = x1r + x2i;
        a[offa + 31] = x1i - x2r;
    }

    public static void cftf162(FloatLargeArray a, long offa, FloatLargeArray w, long startw)
    {
        float wn4r, wk1r, wk1i, wk2r, wk2i, wk3r, wk3i, x0r, x0i, x1r, x1i, x2r, x2i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i, y8r, y8i, y9r, y9i, y10r, y10i, y11r, y11i, y12r, y12i, y13r, y13i, y14r, y14i, y15r, y15i;

        wn4r = w.getFloat(startw + 1);
        wk1r = w.getFloat(startw + 4);
        wk1i = w.getFloat(startw + 5);
        wk3r = w.getFloat(startw + 6);
        wk3i = -w.getFloat(startw + 7);
        wk2r = w.getFloat(startw + 8);
        wk2i = w.getFloat(startw + 9);
        x1r = a.getFloat(offa) - a.getFloat(offa + 17);
        x1i = a.getFloat(offa + 1) + a.getFloat(offa + 16);
        x0r = a.getFloat(offa + 8) - a.getFloat(offa + 25);
        x0i = a.getFloat(offa + 9) + a.getFloat(offa + 24);
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        y0r = x1r + x2r;
        y0i = x1i + x2i;
        y4r = x1r - x2r;
        y4i = x1i - x2i;
        x1r = a.getFloat(offa) + a.getFloat(offa + 17);
        x1i = a.getFloat(offa + 1) - a.getFloat(offa + 16);
        x0r = a.getFloat(offa + 8) + a.getFloat(offa + 25);
        x0i = a.getFloat(offa + 9) - a.getFloat(offa + 24);
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        y8r = x1r - x2i;
        y8i = x1i + x2r;
        y12r = x1r + x2i;
        y12i = x1i - x2r;
        x0r = a.getFloat(offa + 2) - a.getFloat(offa + 19);
        x0i = a.getFloat(offa + 3) + a.getFloat(offa + 18);
        x1r = wk1r * x0r - wk1i * x0i;
        x1i = wk1r * x0i + wk1i * x0r;
        x0r = a.getFloat(offa + 10) - a.getFloat(offa + 27);
        x0i = a.getFloat(offa + 11) + a.getFloat(offa + 26);
        x2r = wk3i * x0r - wk3r * x0i;
        x2i = wk3i * x0i + wk3r * x0r;
        y1r = x1r + x2r;
        y1i = x1i + x2i;
        y5r = x1r - x2r;
        y5i = x1i - x2i;
        x0r = a.getFloat(offa + 2) + a.getFloat(offa + 19);
        x0i = a.getFloat(offa + 3) - a.getFloat(offa + 18);
        x1r = wk3r * x0r - wk3i * x0i;
        x1i = wk3r * x0i + wk3i * x0r;
        x0r = a.getFloat(offa + 10) + a.getFloat(offa + 27);
        x0i = a.getFloat(offa + 11) - a.getFloat(offa + 26);
        x2r = wk1r * x0r + wk1i * x0i;
        x2i = wk1r * x0i - wk1i * x0r;
        y9r = x1r - x2r;
        y9i = x1i - x2i;
        y13r = x1r + x2r;
        y13i = x1i + x2i;
        x0r = a.getFloat(offa + 4) - a.getFloat(offa + 21);
        x0i = a.getFloat(offa + 5) + a.getFloat(offa + 20);
        x1r = wk2r * x0r - wk2i * x0i;
        x1i = wk2r * x0i + wk2i * x0r;
        x0r = a.getFloat(offa + 12) - a.getFloat(offa + 29);
        x0i = a.getFloat(offa + 13) + a.getFloat(offa + 28);
        x2r = wk2i * x0r - wk2r * x0i;
        x2i = wk2i * x0i + wk2r * x0r;
        y2r = x1r + x2r;
        y2i = x1i + x2i;
        y6r = x1r - x2r;
        y6i = x1i - x2i;
        x0r = a.getFloat(offa + 4) + a.getFloat(offa + 21);
        x0i = a.getFloat(offa + 5) - a.getFloat(offa + 20);
        x1r = wk2i * x0r - wk2r * x0i;
        x1i = wk2i * x0i + wk2r * x0r;
        x0r = a.getFloat(offa + 12) + a.getFloat(offa + 29);
        x0i = a.getFloat(offa + 13) - a.getFloat(offa + 28);
        x2r = wk2r * x0r - wk2i * x0i;
        x2i = wk2r * x0i + wk2i * x0r;
        y10r = x1r - x2r;
        y10i = x1i - x2i;
        y14r = x1r + x2r;
        y14i = x1i + x2i;
        x0r = a.getFloat(offa + 6) - a.getFloat(offa + 23);
        x0i = a.getFloat(offa + 7) + a.getFloat(offa + 22);
        x1r = wk3r * x0r - wk3i * x0i;
        x1i = wk3r * x0i + wk3i * x0r;
        x0r = a.getFloat(offa + 14) - a.getFloat(offa + 31);
        x0i = a.getFloat(offa + 15) + a.getFloat(offa + 30);
        x2r = wk1i * x0r - wk1r * x0i;
        x2i = wk1i * x0i + wk1r * x0r;
        y3r = x1r + x2r;
        y3i = x1i + x2i;
        y7r = x1r - x2r;
        y7i = x1i - x2i;
        x0r = a.getFloat(offa + 6) + a.getFloat(offa + 23);
        x0i = a.getFloat(offa + 7) - a.getFloat(offa + 22);
        x1r = wk1i * x0r + wk1r * x0i;
        x1i = wk1i * x0i - wk1r * x0r;
        x0r = a.getFloat(offa + 14) + a.getFloat(offa + 31);
        x0i = a.getFloat(offa + 15) - a.getFloat(offa + 30);
        x2r = wk3i * x0r - wk3r * x0i;
        x2i = wk3i * x0i + wk3r * x0r;
        y11r = x1r + x2r;
        y11i = x1i + x2i;
        y15r = x1r - x2r;
        y15i = x1i - x2i;
        x1r = y0r + y2r;
        x1i = y0i + y2i;
        x2r = y1r + y3r;
        x2i = y1i + y3i;
        a.setFloat(offa, x1r + x2r);
        a.setFloat(offa + 1, x1i + x2i);
        a.setFloat(offa + 2, x1r - x2r);
        a.setFloat(offa + 3, x1i - x2i);
        x1r = y0r - y2r;
        x1i = y0i - y2i;
        x2r = y1r - y3r;
        x2i = y1i - y3i;
        a.setFloat(offa + 4, x1r - x2i);
        a.setFloat(offa + 5, x1i + x2r);
        a.setFloat(offa + 6, x1r + x2i);
        a.setFloat(offa + 7, x1i - x2r);
        x1r = y4r - y6i;
        x1i = y4i + y6r;
        x0r = y5r - y7i;
        x0i = y5i + y7r;
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        a.setFloat(offa + 8, x1r + x2r);
        a.setFloat(offa + 9, x1i + x2i);
        a.setFloat(offa + 10, x1r - x2r);
        a.setFloat(offa + 11, x1i - x2i);
        x1r = y4r + y6i;
        x1i = y4i - y6r;
        x0r = y5r + y7i;
        x0i = y5i - y7r;
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        a.setFloat(offa + 12, x1r - x2i);
        a.setFloat(offa + 13, x1i + x2r);
        a.setFloat(offa + 14, x1r + x2i);
        a.setFloat(offa + 15, x1i - x2r);
        x1r = y8r + y10r;
        x1i = y8i + y10i;
        x2r = y9r - y11r;
        x2i = y9i - y11i;
        a.setFloat(offa + 16, x1r + x2r);
        a.setFloat(offa + 17, x1i + x2i);
        a.setFloat(offa + 18, x1r - x2r);
        a.setFloat(offa + 19, x1i - x2i);
        x1r = y8r - y10r;
        x1i = y8i - y10i;
        x2r = y9r + y11r;
        x2i = y9i + y11i;
        a.setFloat(offa + 20, x1r - x2i);
        a.setFloat(offa + 21, x1i + x2r);
        a.setFloat(offa + 22, x1r + x2i);
        a.setFloat(offa + 23, x1i - x2r);
        x1r = y12r - y14i;
        x1i = y12i + y14r;
        x0r = y13r + y15i;
        x0i = y13i - y15r;
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        a.setFloat(offa + 24, x1r + x2r);
        a.setFloat(offa + 25, x1i + x2i);
        a.setFloat(offa + 26, x1r - x2r);
        a.setFloat(offa + 27, x1i - x2i);
        x1r = y12r + y14i;
        x1i = y12i - y14r;
        x0r = y13r - y15i;
        x0i = y13i + y15r;
        x2r = wn4r * (x0r - x0i);
        x2i = wn4r * (x0i + x0r);
        a.setFloat(offa + 28, x1r - x2i);
        a.setFloat(offa + 29, x1i + x2r);
        a.setFloat(offa + 30, x1r + x2i);
        a.setFloat(offa + 31, x1i - x2r);
    }

    public static void cftf081(float[] a, int offa, float[] w, int startw)
    {
        float wn4r, x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;

        wn4r = w[startw + 1];
        x0r = a[offa] + a[offa + 8];
        x0i = a[offa + 1] + a[offa + 9];
        x1r = a[offa] - a[offa + 8];
        x1i = a[offa + 1] - a[offa + 9];
        x2r = a[offa + 4] + a[offa + 12];
        x2i = a[offa + 5] + a[offa + 13];
        x3r = a[offa + 4] - a[offa + 12];
        x3i = a[offa + 5] - a[offa + 13];
        y0r = x0r + x2r;
        y0i = x0i + x2i;
        y2r = x0r - x2r;
        y2i = x0i - x2i;
        y1r = x1r - x3i;
        y1i = x1i + x3r;
        y3r = x1r + x3i;
        y3i = x1i - x3r;
        x0r = a[offa + 2] + a[offa + 10];
        x0i = a[offa + 3] + a[offa + 11];
        x1r = a[offa + 2] - a[offa + 10];
        x1i = a[offa + 3] - a[offa + 11];
        x2r = a[offa + 6] + a[offa + 14];
        x2i = a[offa + 7] + a[offa + 15];
        x3r = a[offa + 6] - a[offa + 14];
        x3i = a[offa + 7] - a[offa + 15];
        y4r = x0r + x2r;
        y4i = x0i + x2i;
        y6r = x0r - x2r;
        y6i = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        x2r = x1r + x3i;
        x2i = x1i - x3r;
        y5r = wn4r * (x0r - x0i);
        y5i = wn4r * (x0r + x0i);
        y7r = wn4r * (x2r - x2i);
        y7i = wn4r * (x2r + x2i);
        a[offa + 8] = y1r + y5r;
        a[offa + 9] = y1i + y5i;
        a[offa + 10] = y1r - y5r;
        a[offa + 11] = y1i - y5i;
        a[offa + 12] = y3r - y7i;
        a[offa + 13] = y3i + y7r;
        a[offa + 14] = y3r + y7i;
        a[offa + 15] = y3i - y7r;
        a[offa] = y0r + y4r;
        a[offa + 1] = y0i + y4i;
        a[offa + 2] = y0r - y4r;
        a[offa + 3] = y0i - y4i;
        a[offa + 4] = y2r - y6i;
        a[offa + 5] = y2i + y6r;
        a[offa + 6] = y2r + y6i;
        a[offa + 7] = y2i - y6r;
    }

    public static void cftf081(FloatLargeArray a, long offa, FloatLargeArray w, long startw)
    {
        float wn4r, x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;

        wn4r = w.getFloat(startw + 1);
        x0r = a.getFloat(offa) + a.getFloat(offa + 8);
        x0i = a.getFloat(offa + 1) + a.getFloat(offa + 9);
        x1r = a.getFloat(offa) - a.getFloat(offa + 8);
        x1i = a.getFloat(offa + 1) - a.getFloat(offa + 9);
        x2r = a.getFloat(offa + 4) + a.getFloat(offa + 12);
        x2i = a.getFloat(offa + 5) + a.getFloat(offa + 13);
        x3r = a.getFloat(offa + 4) - a.getFloat(offa + 12);
        x3i = a.getFloat(offa + 5) - a.getFloat(offa + 13);
        y0r = x0r + x2r;
        y0i = x0i + x2i;
        y2r = x0r - x2r;
        y2i = x0i - x2i;
        y1r = x1r - x3i;
        y1i = x1i + x3r;
        y3r = x1r + x3i;
        y3i = x1i - x3r;
        x0r = a.getFloat(offa + 2) + a.getFloat(offa + 10);
        x0i = a.getFloat(offa + 3) + a.getFloat(offa + 11);
        x1r = a.getFloat(offa + 2) - a.getFloat(offa + 10);
        x1i = a.getFloat(offa + 3) - a.getFloat(offa + 11);
        x2r = a.getFloat(offa + 6) + a.getFloat(offa + 14);
        x2i = a.getFloat(offa + 7) + a.getFloat(offa + 15);
        x3r = a.getFloat(offa + 6) - a.getFloat(offa + 14);
        x3i = a.getFloat(offa + 7) - a.getFloat(offa + 15);
        y4r = x0r + x2r;
        y4i = x0i + x2i;
        y6r = x0r - x2r;
        y6i = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        x2r = x1r + x3i;
        x2i = x1i - x3r;
        y5r = wn4r * (x0r - x0i);
        y5i = wn4r * (x0r + x0i);
        y7r = wn4r * (x2r - x2i);
        y7i = wn4r * (x2r + x2i);
        a.setFloat(offa + 8, y1r + y5r);
        a.setFloat(offa + 9, y1i + y5i);
        a.setFloat(offa + 10, y1r - y5r);
        a.setFloat(offa + 11, y1i - y5i);
        a.setFloat(offa + 12, y3r - y7i);
        a.setFloat(offa + 13, y3i + y7r);
        a.setFloat(offa + 14, y3r + y7i);
        a.setFloat(offa + 15, y3i - y7r);
        a.setFloat(offa, y0r + y4r);
        a.setFloat(offa + 1, y0i + y4i);
        a.setFloat(offa + 2, y0r - y4r);
        a.setFloat(offa + 3, y0i - y4i);
        a.setFloat(offa + 4, y2r - y6i);
        a.setFloat(offa + 5, y2i + y6r);
        a.setFloat(offa + 6, y2r + y6i);
        a.setFloat(offa + 7, y2i - y6r);
    }

    public static void cftf082(float[] a, int offa, float[] w, int startw)
    {
        float wn4r, wk1r, wk1i, x0r, x0i, x1r, x1i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;

        wn4r = w[startw + 1];
        wk1r = w[startw + 2];
        wk1i = w[startw + 3];
        y0r = a[offa] - a[offa + 9];
        y0i = a[offa + 1] + a[offa + 8];
        y1r = a[offa] + a[offa + 9];
        y1i = a[offa + 1] - a[offa + 8];
        x0r = a[offa + 4] - a[offa + 13];
        x0i = a[offa + 5] + a[offa + 12];
        y2r = wn4r * (x0r - x0i);
        y2i = wn4r * (x0i + x0r);
        x0r = a[offa + 4] + a[offa + 13];
        x0i = a[offa + 5] - a[offa + 12];
        y3r = wn4r * (x0r - x0i);
        y3i = wn4r * (x0i + x0r);
        x0r = a[offa + 2] - a[offa + 11];
        x0i = a[offa + 3] + a[offa + 10];
        y4r = wk1r * x0r - wk1i * x0i;
        y4i = wk1r * x0i + wk1i * x0r;
        x0r = a[offa + 2] + a[offa + 11];
        x0i = a[offa + 3] - a[offa + 10];
        y5r = wk1i * x0r - wk1r * x0i;
        y5i = wk1i * x0i + wk1r * x0r;
        x0r = a[offa + 6] - a[offa + 15];
        x0i = a[offa + 7] + a[offa + 14];
        y6r = wk1i * x0r - wk1r * x0i;
        y6i = wk1i * x0i + wk1r * x0r;
        x0r = a[offa + 6] + a[offa + 15];
        x0i = a[offa + 7] - a[offa + 14];
        y7r = wk1r * x0r - wk1i * x0i;
        y7i = wk1r * x0i + wk1i * x0r;
        x0r = y0r + y2r;
        x0i = y0i + y2i;
        x1r = y4r + y6r;
        x1i = y4i + y6i;
        a[offa] = x0r + x1r;
        a[offa + 1] = x0i + x1i;
        a[offa + 2] = x0r - x1r;
        a[offa + 3] = x0i - x1i;
        x0r = y0r - y2r;
        x0i = y0i - y2i;
        x1r = y4r - y6r;
        x1i = y4i - y6i;
        a[offa + 4] = x0r - x1i;
        a[offa + 5] = x0i + x1r;
        a[offa + 6] = x0r + x1i;
        a[offa + 7] = x0i - x1r;
        x0r = y1r - y3i;
        x0i = y1i + y3r;
        x1r = y5r - y7r;
        x1i = y5i - y7i;
        a[offa + 8] = x0r + x1r;
        a[offa + 9] = x0i + x1i;
        a[offa + 10] = x0r - x1r;
        a[offa + 11] = x0i - x1i;
        x0r = y1r + y3i;
        x0i = y1i - y3r;
        x1r = y5r + y7r;
        x1i = y5i + y7i;
        a[offa + 12] = x0r - x1i;
        a[offa + 13] = x0i + x1r;
        a[offa + 14] = x0r + x1i;
        a[offa + 15] = x0i - x1r;
    }

    public static void cftf082(FloatLargeArray a, long offa, FloatLargeArray w, long startw)
    {
        float wn4r, wk1r, wk1i, x0r, x0i, x1r, x1i, y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;

        wn4r = w.getFloat(startw + 1);
        wk1r = w.getFloat(startw + 2);
        wk1i = w.getFloat(startw + 3);
        y0r = a.getFloat(offa) - a.getFloat(offa + 9);
        y0i = a.getFloat(offa + 1) + a.getFloat(offa + 8);
        y1r = a.getFloat(offa) + a.getFloat(offa + 9);
        y1i = a.getFloat(offa + 1) - a.getFloat(offa + 8);
        x0r = a.getFloat(offa + 4) - a.getFloat(offa + 13);
        x0i = a.getFloat(offa + 5) + a.getFloat(offa + 12);
        y2r = wn4r * (x0r - x0i);
        y2i = wn4r * (x0i + x0r);
        x0r = a.getFloat(offa + 4) + a.getFloat(offa + 13);
        x0i = a.getFloat(offa + 5) - a.getFloat(offa + 12);
        y3r = wn4r * (x0r - x0i);
        y3i = wn4r * (x0i + x0r);
        x0r = a.getFloat(offa + 2) - a.getFloat(offa + 11);
        x0i = a.getFloat(offa + 3) + a.getFloat(offa + 10);
        y4r = wk1r * x0r - wk1i * x0i;
        y4i = wk1r * x0i + wk1i * x0r;
        x0r = a.getFloat(offa + 2) + a.getFloat(offa + 11);
        x0i = a.getFloat(offa + 3) - a.getFloat(offa + 10);
        y5r = wk1i * x0r - wk1r * x0i;
        y5i = wk1i * x0i + wk1r * x0r;
        x0r = a.getFloat(offa + 6) - a.getFloat(offa + 15);
        x0i = a.getFloat(offa + 7) + a.getFloat(offa + 14);
        y6r = wk1i * x0r - wk1r * x0i;
        y6i = wk1i * x0i + wk1r * x0r;
        x0r = a.getFloat(offa + 6) + a.getFloat(offa + 15);
        x0i = a.getFloat(offa + 7) - a.getFloat(offa + 14);
        y7r = wk1r * x0r - wk1i * x0i;
        y7i = wk1r * x0i + wk1i * x0r;
        x0r = y0r + y2r;
        x0i = y0i + y2i;
        x1r = y4r + y6r;
        x1i = y4i + y6i;
        a.setFloat(offa, x0r + x1r);
        a.setFloat(offa + 1, x0i + x1i);
        a.setFloat(offa + 2, x0r - x1r);
        a.setFloat(offa + 3, x0i - x1i);
        x0r = y0r - y2r;
        x0i = y0i - y2i;
        x1r = y4r - y6r;
        x1i = y4i - y6i;
        a.setFloat(offa + 4, x0r - x1i);
        a.setFloat(offa + 5, x0i + x1r);
        a.setFloat(offa + 6, x0r + x1i);
        a.setFloat(offa + 7, x0i - x1r);
        x0r = y1r - y3i;
        x0i = y1i + y3r;
        x1r = y5r - y7r;
        x1i = y5i - y7i;
        a.setFloat(offa + 8, x0r + x1r);
        a.setFloat(offa + 9, x0i + x1i);
        a.setFloat(offa + 10, x0r - x1r);
        a.setFloat(offa + 11, x0i - x1i);
        x0r = y1r + y3i;
        x0i = y1i - y3r;
        x1r = y5r + y7r;
        x1i = y5i + y7i;
        a.setFloat(offa + 12, x0r - x1i);
        a.setFloat(offa + 13, x0i + x1r);
        a.setFloat(offa + 14, x0r + x1i);
        a.setFloat(offa + 15, x0i - x1r);
    }

    public static void cftf040(float[] a, int offa)
    {
        float x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;

        x0r = a[offa] + a[offa + 4];
        x0i = a[offa + 1] + a[offa + 5];
        x1r = a[offa] - a[offa + 4];
        x1i = a[offa + 1] - a[offa + 5];
        x2r = a[offa + 2] + a[offa + 6];
        x2i = a[offa + 3] + a[offa + 7];
        x3r = a[offa + 2] - a[offa + 6];
        x3i = a[offa + 3] - a[offa + 7];
        a[offa] = x0r + x2r;
        a[offa + 1] = x0i + x2i;
        a[offa + 2] = x1r - x3i;
        a[offa + 3] = x1i + x3r;
        a[offa + 4] = x0r - x2r;
        a[offa + 5] = x0i - x2i;
        a[offa + 6] = x1r + x3i;
        a[offa + 7] = x1i - x3r;
    }

    public static void cftf040(FloatLargeArray a, long offa)
    {
        float x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;

        x0r = a.getFloat(offa) + a.getFloat(offa + 4);
        x0i = a.getFloat(offa + 1) + a.getFloat(offa + 5);
        x1r = a.getFloat(offa) - a.getFloat(offa + 4);
        x1i = a.getFloat(offa + 1) - a.getFloat(offa + 5);
        x2r = a.getFloat(offa + 2) + a.getFloat(offa + 6);
        x2i = a.getFloat(offa + 3) + a.getFloat(offa + 7);
        x3r = a.getFloat(offa + 2) - a.getFloat(offa + 6);
        x3i = a.getFloat(offa + 3) - a.getFloat(offa + 7);
        a.setFloat(offa, x0r + x2r);
        a.setFloat(offa + 1, x0i + x2i);
        a.setFloat(offa + 2, x1r - x3i);
        a.setFloat(offa + 3, x1i + x3r);
        a.setFloat(offa + 4, x0r - x2r);
        a.setFloat(offa + 5, x0i - x2i);
        a.setFloat(offa + 6, x1r + x3i);
        a.setFloat(offa + 7, x1i - x3r);
    }

    public static void cftb040(float[] a, int offa)
    {
        float x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;

        x0r = a[offa] + a[offa + 4];
        x0i = a[offa + 1] + a[offa + 5];
        x1r = a[offa] - a[offa + 4];
        x1i = a[offa + 1] - a[offa + 5];
        x2r = a[offa + 2] + a[offa + 6];
        x2i = a[offa + 3] + a[offa + 7];
        x3r = a[offa + 2] - a[offa + 6];
        x3i = a[offa + 3] - a[offa + 7];
        a[offa] = x0r + x2r;
        a[offa + 1] = x0i + x2i;
        a[offa + 2] = x1r + x3i;
        a[offa + 3] = x1i - x3r;
        a[offa + 4] = x0r - x2r;
        a[offa + 5] = x0i - x2i;
        a[offa + 6] = x1r - x3i;
        a[offa + 7] = x1i + x3r;
    }

    public static void cftb040(FloatLargeArray a, long offa)
    {
        float x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;

        x0r = a.getFloat(offa) + a.getFloat(offa + 4);
        x0i = a.getFloat(offa + 1) + a.getFloat(offa + 5);
        x1r = a.getFloat(offa) - a.getFloat(offa + 4);
        x1i = a.getFloat(offa + 1) - a.getFloat(offa + 5);
        x2r = a.getFloat(offa + 2) + a.getFloat(offa + 6);
        x2i = a.getFloat(offa + 3) + a.getFloat(offa + 7);
        x3r = a.getFloat(offa + 2) - a.getFloat(offa + 6);
        x3i = a.getFloat(offa + 3) - a.getFloat(offa + 7);
        a.setFloat(offa, x0r + x2r);
        a.setFloat(offa + 1, x0i + x2i);
        a.setFloat(offa + 2, x1r + x3i);
        a.setFloat(offa + 3, x1i - x3r);
        a.setFloat(offa + 4, x0r - x2r);
        a.setFloat(offa + 5, x0i - x2i);
        a.setFloat(offa + 6, x1r - x3i);
        a.setFloat(offa + 7, x1i + x3r);
    }

    public static void cftx020(float[] a, int offa)
    {
        float x0r, x0i;
        x0r = a[offa] - a[offa + 2];
        x0i = -a[offa + 1] + a[offa + 3];
        a[offa] += a[offa + 2];
        a[offa + 1] += a[offa + 3];
        a[offa + 2] = x0r;
        a[offa + 3] = x0i;
    }

    public static void cftx020(FloatLargeArray a, long offa)
    {
        float x0r, x0i;
        x0r = a.getFloat(offa) - a.getFloat(offa + 2);
        x0i = -a.getFloat(offa + 1) + a.getFloat(offa + 3);
        a.setFloat(offa, a.getFloat(offa) + a.getFloat(offa + 2));
        a.setFloat(offa + 1, a.getFloat(offa + 1) + a.getFloat(offa + 3));
        a.setFloat(offa + 2, x0r);
        a.setFloat(offa + 3, x0i);
    }

    public static void cftxb020(float[] a, int offa)
    {
        float x0r, x0i;

        x0r = a[offa] - a[offa + 2];
        x0i = a[offa + 1] - a[offa + 3];
        a[offa] += a[offa + 2];
        a[offa + 1] += a[offa + 3];
        a[offa + 2] = x0r;
        a[offa + 3] = x0i;
    }

    public static void cftxb020(FloatLargeArray a, long offa)
    {
        float x0r, x0i;

        x0r = a.getFloat(offa) - a.getFloat(offa + 2);
        x0i = a.getFloat(offa + 1) - a.getFloat(offa + 3);
        a.setFloat(offa, a.getFloat(offa) + a.getFloat(offa + 2));
        a.setFloat(offa + 1, a.getFloat(offa + 1) + a.getFloat(offa + 3));
        a.setFloat(offa + 2, x0r);
        a.setFloat(offa + 3, x0i);
    }

    public static void cftxc020(float[] a, int offa)
    {
        float x0r, x0i;
        x0r = a[offa] - a[offa + 2];
        x0i = a[offa + 1] + a[offa + 3];
        a[offa] += a[offa + 2];
        a[offa + 1] -= a[offa + 3];
        a[offa + 2] = x0r;
        a[offa + 3] = x0i;
    }

    public static void cftxc020(FloatLargeArray a, long offa)
    {
        float x0r, x0i;
        x0r = a.getFloat(offa) - a.getFloat(offa + 2);
        x0i = a.getFloat(offa + 1) + a.getFloat(offa + 3);
        a.setFloat(offa, a.getFloat(offa) + a.getFloat(offa + 2));
        a.setFloat(offa + 1, a.getFloat(offa + 1) - a.getFloat(offa + 3));
        a.setFloat(offa + 2, x0r);
        a.setFloat(offa + 3, x0i);
    }

    public static void rftfsub(int n, float[] a, int offa, int nc, float[] c, int startc)
    {
        int k, kk, ks, m;
        float wkr, wki, xr, xi, yr, yi;
        int idx1, idx2;

        m = n >> 1;
        ks = 2 * nc / m;
        kk = 0;
        for (int j = 2; j < m; j += 2) {
            k = n - j;
            kk += ks;
            wkr = 0.5f - c[startc + nc - kk];
            wki = c[startc + kk];
            idx1 = offa + j;
            idx2 = offa + k;
            xr = a[idx1] - a[idx2];
            xi = a[idx1 + 1] + a[idx2 + 1];
            yr = wkr * xr - wki * xi;
            yi = wkr * xi + wki * xr;
            a[idx1] -= yr;
            a[idx1 + 1] = yi - a[idx1 + 1];
            a[idx2] += yr;
            a[idx2 + 1] = yi - a[idx2 + 1];
        }
        a[offa + m + 1] = -a[offa + m + 1];
    }

    public static void rftfsub(long n, FloatLargeArray a, long offa, long nc, FloatLargeArray c, long startc)
    {
        long k, kk, ks, m;
        float wkr, wki, xr, xi, yr, yi;
        long idx1, idx2;

        m = n >> 1l;
        ks = 2 * nc / m;
        kk = 0;
        for (long j = 2; j < m; j += 2) {
            k = n - j;
            kk += ks;
            wkr = 0.5f - c.getFloat(startc + nc - kk);
            wki = c.getFloat(startc + kk);
            idx1 = offa + j;
            idx2 = offa + k;
            xr = a.getFloat(idx1) - a.getFloat(idx2);
            xi = a.getFloat(idx1 + 1) + a.getFloat(idx2 + 1);
            yr = wkr * xr - wki * xi;
            yi = wkr * xi + wki * xr;
            a.setFloat(idx1, a.getFloat(idx1) - yr);
            a.setFloat(idx1 + 1, yi - a.getFloat(idx1 + 1));
            a.setFloat(idx2, a.getFloat(idx2) + yr);
            a.setFloat(idx2 + 1, yi - a.getFloat(idx2 + 1));
        }
        a.setFloat(offa + m + 1, -a.getFloat(offa + m + 1));
    }

    public static void rftbsub(int n, float[] a, int offa, int nc, float[] c, int startc)
    {
        int k, kk, ks, m;
        float wkr, wki, xr, xi, yr, yi;
        int idx1, idx2;

        m = n >> 1;
        ks = 2 * nc / m;
        kk = 0;
        for (int j = 2; j < m; j += 2) {
            k = n - j;
            kk += ks;
            wkr = 0.5f - c[startc + nc - kk];
            wki = c[startc + kk];
            idx1 = offa + j;
            idx2 = offa + k;
            xr = a[idx1] - a[idx2];
            xi = a[idx1 + 1] + a[idx2 + 1];
            yr = wkr * xr - wki * xi;
            yi = wkr * xi + wki * xr;
            a[idx1] -= yr;
            a[idx1 + 1] -= yi;
            a[idx2] += yr;
            a[idx2 + 1] -= yi;
        }
    }

    public static void rftbsub(long n, FloatLargeArray a, long offa, long nc, FloatLargeArray c, long startc)
    {
        long k, kk, ks, m;
        float wkr, wki, xr, xi, yr, yi;
        long idx1, idx2;

        m = n >> 1;
        ks = 2 * nc / m;
        kk = 0;
        for (long j = 2; j < m; j += 2) {
            k = n - j;
            kk += ks;
            wkr = 0.5f - c.getFloat(startc + nc - kk);
            wki = c.getFloat(startc + kk);
            idx1 = offa + j;
            idx2 = offa + k;
            xr = a.getFloat(idx1) - a.getFloat(idx2);
            xi = a.getFloat(idx1 + 1) + a.getFloat(idx2 + 1);
            yr = wkr * xr - wki * xi;
            yi = wkr * xi + wki * xr;
            a.setFloat(idx1, a.getFloat(idx1) - yr);
            a.setFloat(idx1 + 1, a.getFloat(idx1 + 1) - yi);
            a.setFloat(idx2, a.getFloat(idx2) + yr);
            a.setFloat(idx2 + 1, a.getFloat(idx2 + 1) - yi);
        }
    }

    public static void dctsub(int n, float[] a, int offa, int nc, float[] c, int startc)
    {
        int k, kk, ks, m;
        float wkr, wki, xr;
        int idx0, idx1, idx2;

        m = n >> 1;
        ks = nc / n;
        kk = 0;
        for (int j = 1; j < m; j++) {
            k = n - j;
            kk += ks;
            idx0 = startc + kk;
            idx1 = offa + j;
            idx2 = offa + k;
            wkr = c[idx0] - c[startc + nc - kk];
            wki = c[idx0] + c[startc + nc - kk];
            xr = wki * a[idx1] - wkr * a[idx2];
            a[idx1] = wkr * a[idx1] + wki * a[idx2];
            a[idx2] = xr;
        }
        a[offa + m] *= c[startc];
    }

    public static void dctsub(long n, FloatLargeArray a, long offa, long nc, FloatLargeArray c, long startc)
    {
        long k, kk, ks, m;
        float wkr, wki, xr;
        long idx0, idx1, idx2;

        m = n >> 1l;
        ks = nc / n;
        kk = 0;
        for (long j = 1; j < m; j++) {
            k = n - j;
            kk += ks;
            idx0 = startc + kk;
            idx1 = offa + j;
            idx2 = offa + k;
            wkr = c.getFloat(idx0) - c.getFloat(startc + nc - kk);
            wki = c.getFloat(idx0) + c.getFloat(startc + nc - kk);
            xr = wki * a.getFloat(idx1) - wkr * a.getFloat(idx2);
            a.setFloat(idx1, wkr * a.getFloat(idx1) + wki * a.getFloat(idx2));
            a.setFloat(idx2, xr);
        }
        a.setFloat(offa + m, a.getFloat(offa + m) * c.getFloat(startc));
    }

    public static void scale(final int n, final double m, final double[] a, final int offa, boolean complex)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        int n2;
        if (complex) {
            n2 = 2 * n;
        } else {
            n2 = n;
        }
        if ((nthreads > 1) && (n2 > CommonUtils.getThreadsBeginN_1D_FFT_2Threads())) {
            nthreads = 2;
            final int k = n2 / nthreads;
            Future<?>[] futures = new Future[nthreads];
            for (int i = 0; i < nthreads; i++) {
                final int firstIdx = offa + i * k;
                final int lastIdx = (i == (nthreads - 1)) ? offa + n2 : firstIdx + k;
                futures[i] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            a[i] *= m;
                        }

                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(CommonUtils.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(CommonUtils.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {
            int firstIdx = offa;
            int lastIdx = offa + n2;
            for (int i = firstIdx; i < lastIdx; i++) {
                a[i] *= m;
            }
        }
    }

    public static void scale(final long nl, final double m, final DoubleLargeArray a, long offa, boolean complex)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        long n2;
        if (complex) {
            n2 = 2 * nl;
        } else {
            n2 = nl;
        }
        if ((nthreads > 1) && (n2 >= CommonUtils.getThreadsBeginN_1D_FFT_2Threads())) {
            final long k = n2 / nthreads;
            Future<?>[] futures = new Future[nthreads];
            for (int i = 0; i < nthreads; i++) {
                final long firstIdx = offa + i * k;
                final long lastIdx = (i == (nthreads - 1)) ? offa + n2 : firstIdx + k;
                futures[i] = ConcurrencyUtils.submit(new Runnable()
                {

                    public void run()
                    {
                        for (long i = firstIdx; i < lastIdx; i++) {
                            a.setDouble(i, a.getDouble(i) * m);
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(CommonUtils.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(CommonUtils.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {
            for (long i = offa; i < offa + n2; i++) {
                a.setDouble(i, a.getDouble(i) * m);
            }

        }
    }

    public static void scale(final int n, final float m, final float[] a, final int offa, boolean complex)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        int n2;
        if (complex) {
            n2 = 2 * n;
        } else {
            n2 = n;
        }
        if ((nthreads > 1) && (n2 > CommonUtils.getThreadsBeginN_1D_FFT_2Threads())) {
            nthreads = 2;
            final int k = n2 / nthreads;
            Future<?>[] futures = new Future[nthreads];
            for (int i = 0; i < nthreads; i++) {
                final int firstIdx = offa + i * k;
                final int lastIdx = (i == (nthreads - 1)) ? offa + n2 : firstIdx + k;
                futures[i] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (int i = firstIdx; i < lastIdx; i++) {
                            a[i] *= m;
                        }

                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(CommonUtils.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(CommonUtils.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {
            int firstIdx = offa;
            int lastIdx = offa + n2;
            for (int i = firstIdx; i < lastIdx; i++) {
                a[i] *= m;
            }
        }
    }

    public static void scale(final long nl, final float m, final FloatLargeArray a, long offa, boolean complex)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        long n2;
        if (complex) {
            n2 = 2 * nl;
        } else {
            n2 = nl;
        }
        if ((nthreads > 1) && (n2 >= CommonUtils.getThreadsBeginN_1D_FFT_2Threads())) {
            final long k = n2 / nthreads;
            Future<?>[] futures = new Future[nthreads];
            for (int i = 0; i < nthreads; i++) {
                final long firstIdx = offa + i * k;
                final long lastIdx = (i == (nthreads - 1)) ? offa + n2 : firstIdx + k;
                futures[i] = ConcurrencyUtils.submit(new Runnable()
                {

                    public void run()
                    {
                        for (long i = firstIdx; i < lastIdx; i++) {
                            a.setDouble(i, a.getDouble(i) * m);
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(CommonUtils.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(CommonUtils.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {
            for (long i = offa; i < offa + n2; i++) {
                a.setDouble(i, a.getDouble(i) * m);
            }

        }
    }

}
