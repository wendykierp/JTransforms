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
package org.jtransforms.dct;

import java.util.concurrent.Future;
import org.jtransforms.fft.FloatFFT_1D;
import org.jtransforms.utils.CommonUtils;
import java.util.concurrent.ExecutionException;
import java.util.logging.Level;
import java.util.logging.Logger;
import pl.edu.icm.jlargearrays.ConcurrencyUtils;
import pl.edu.icm.jlargearrays.FloatLargeArray;
import pl.edu.icm.jlargearrays.LargeArray;
import pl.edu.icm.jlargearrays.LongLargeArray;
import pl.edu.icm.jlargearrays.LargeArrayUtils;
import static org.apache.commons.math3.util.FastMath.*;

/**
 * Computes 1D Discrete Cosine Transform (DCT) of single precision data. The
 * size of data can be an arbitrary number. This is a parallel implementation of
 * split-radix and mixed-radix algorithms optimized for SMP systems. <br>
 * <br>
 * Part of the code is derived from General Purpose FFT Package written by Takuya Ooura
 * (http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html)
 *  
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class FloatDCT_1D
{

    private int n;

    private long nl;

    private int[] ip;

    private LongLargeArray ipl;

    private float[] w;

    private FloatLargeArray wl;

    private int nw;

    private long nwl;

    private int nc;

    private long ncl;

    private boolean isPowerOfTwo = false;

    private FloatFFT_1D fft;

    private static final float PI = 3.14159265358979311599796346854418516f;

    private boolean useLargeArrays;

    /**
     * Creates new instance of FloatDCT_1D.
     *  
     * @param n size of data
     *
     */
    public FloatDCT_1D(long n)
    {
        if (n < 1) {
            throw new IllegalArgumentException("n must be greater than 0");
        }
        this.useLargeArrays = (CommonUtils.isUseLargeArrays() || n > LargeArray.getMaxSizeOf32bitArray());

        this.n = (int) n;
        this.nl = n;
        if (!useLargeArrays) {
            if (n > (1 << 28)) {
                throw new IllegalArgumentException("n must be smaller or equal to " + (1 << 28) + " when useLargeArrays argument is set to false");
            }
            if (CommonUtils.isPowerOf2(n)) {
                this.isPowerOfTwo = true;
                this.ip = new int[(int) ceil(2 + (1 << (int) (log(n / 2 + 0.5) / log(2)) / 2))];
                this.w = new float[this.n * 5 / 4];
                nw = ip[0];
                if (n > (nw << 2)) {
                    nw = this.n >> 2;
                    CommonUtils.makewt(nw, ip, w);
                }
                nc = ip[1];
                if (n > nc) {
                    nc = this.n;
                    CommonUtils.makect(nc, w, nw, ip);
                }
            } else {
                this.w = makect(this.n);
                fft = new FloatFFT_1D(2 * n);
            }
        } else if (CommonUtils.isPowerOf2(n)) {
            this.isPowerOfTwo = true;
            this.ipl = new LongLargeArray((long) ceil(2 + (1l << (long) (log(n / 2 + 0.5) / log(2)) / 2)));
            this.wl = new FloatLargeArray(this.nl * 5l / 4l);
            nwl = ipl.getLong(0);
            if (n > (nwl << 2l)) {
                nwl = this.nl >> 2l;
                CommonUtils.makewt(nwl, ipl, wl);
            }
            ncl = ipl.getLong(1);
            if (n > ncl) {
                ncl = this.nl;
                CommonUtils.makect(ncl, wl, nwl, ipl);
            }
        } else {
            this.wl = makect(n);
            fft = new FloatFFT_1D(2 * n);
        }

    }

    /**
     * Computes 1D forward DCT (DCT-II) leaving the result in <code>a</code>.
     *  
     * @param a
     *              data to transform
     * @param scale
     *              if true then scaling is performed
     */
    public void forward(float[] a, boolean scale)
    {
        forward(a, 0, scale);
    }

    /**
     * Computes 1D forward DCT (DCT-II) leaving the result in <code>a</code>.
     *  
     * @param a
     *              data to transform
     * @param scale
     *              if true then scaling is performed
     */
    public void forward(FloatLargeArray a, boolean scale)
    {
        forward(a, 0, scale);
    }

    /**
     * Computes 1D forward DCT (DCT-II) leaving the result in <code>a</code>.
     *  
     * @param a
     *              data to transform
     * @param offa
     *              index of the first element in array <code>a</code>
     * @param scale
     *              if true then scaling is performed
     */
    public void forward(final float[] a, final int offa, boolean scale)
    {
        if (n == 1)
            return;
        if (useLargeArrays) {
            forward(new FloatLargeArray(a), offa, scale);
        } else if (isPowerOfTwo) {
            float xr = a[offa + n - 1];
            for (int j = n - 2; j >= 2; j -= 2) {
                a[offa + j + 1] = a[offa + j] - a[offa + j - 1];
                a[offa + j] += a[offa + j - 1];
            }
            a[offa + 1] = a[offa] - xr;
            a[offa] += xr;
            if (n > 4) {
                rftbsub(n, a, offa, nc, w, nw);
                CommonUtils.cftbsub(n, a, offa, ip, nw, w);
            } else if (n == 4) {
                CommonUtils.cftbsub(n, a, offa, ip, nw, w);
            }
            CommonUtils.dctsub(n, a, offa, nc, w, nw);
            if (scale) {
                CommonUtils.scale(n, (float) sqrt(2.0 / n), a, offa, false);
                a[offa] = a[offa] / (float) sqrt(2.0);
            }
        } else {
            final int twon = 2 * n;
            final float[] t = new float[twon];
            System.arraycopy(a, offa, t, 0, n);
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            for (int i = n; i < twon; i++) {
                t[i] = t[twon - i - 1];
            }
            fft.realForward(t);
            if ((nthreads > 1) && (n > CommonUtils.getThreadsBeginN_1D_FFT_2Threads())) {
                nthreads = 2;
                final int k = n / nthreads;
                Future<?>[] futures = new Future[nthreads];
                for (int j = 0; j < nthreads; j++) {
                    final int firstIdx = j * k;
                    final int lastIdx = (j == (nthreads - 1)) ? n : firstIdx + k;
                    futures[j] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            for (int i = firstIdx; i < lastIdx; i++) {
                                int twoi = 2 * i;
                                int idx = offa + i;
                                a[idx] = w[twoi] * t[twoi] - w[twoi + 1] * t[twoi + 1];
                            }

                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(FloatDCT_1D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(FloatDCT_1D.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else {
                for (int i = 0; i < n; i++) {
                    int twoi = 2 * i;
                    int idx = offa + i;
                    a[idx] = w[twoi] * t[twoi] - w[twoi + 1] * t[twoi + 1];
                }
            }
            if (scale) {
                CommonUtils.scale(n, 1 / (float) sqrt(twon), a, offa, false);
                a[offa] = a[offa] / (float) sqrt(2.0);
            }
        }
    }

    /**
     * Computes 1D forward DCT (DCT-II) leaving the result in <code>a</code>.
     *  
     * @param a
     *              data to transform
     * @param offa
     *              index of the first element in array <code>a</code>
     * @param scale
     *              if true then scaling is performed
     */
    public void forward(final FloatLargeArray a, final long offa, boolean scale)
    {
        if (nl == 1)
            return;
        if (!useLargeArrays) {
            if (!a.isLarge() && !a.isConstant() && offa < Integer.MAX_VALUE) {
                forward(a.getData(), (int) offa, scale);
            } else {
                throw new IllegalArgumentException("The data array is too big.");
            }
        } else if (isPowerOfTwo) {
            float xr = a.getFloat(offa + nl - 1);
            for (long j = nl - 2; j >= 2; j -= 2) {
                a.setFloat(offa + j + 1, a.getFloat(offa + j) - a.getFloat(offa + j - 1));
                a.setFloat(offa + j, a.getFloat(offa + j) + a.getFloat(offa + j - 1));
            }
            a.setFloat(offa + 1, a.getFloat(offa) - xr);
            a.setFloat(offa, a.getFloat(offa) + xr);
            if (nl > 4) {
                rftbsub(nl, a, offa, ncl, wl, nwl);
                CommonUtils.cftbsub(nl, a, offa, ipl, nwl, wl);
            } else if (nl == 4) {
                CommonUtils.cftbsub(nl, a, offa, ipl, nwl, wl);
            }
            CommonUtils.dctsub(nl, a, offa, ncl, wl, nwl);
            if (scale) {
                CommonUtils.scale(nl, (float) sqrt(2.0 / nl), a, offa, false);
                a.setFloat(offa, a.getFloat(offa) / (float) sqrt(2.0));
            }
        } else {
            final long twon = 2 * nl;
            final FloatLargeArray t = new FloatLargeArray(twon);
            LargeArrayUtils.arraycopy(a, offa, t, 0, nl);
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            for (long i = nl; i < twon; i++) {
                t.setFloat(i, t.getFloat(twon - i - 1));
            }
            fft.realForward(t);
            if ((nthreads > 1) && (nl > CommonUtils.getThreadsBeginN_1D_FFT_2Threads())) {
                nthreads = 2;
                final long k = nl / nthreads;
                Future<?>[] futures = new Future[nthreads];
                for (int j = 0; j < nthreads; j++) {
                    final long firstIdx = j * k;
                    final long lastIdx = (j == (nthreads - 1)) ? nl : firstIdx + k;
                    futures[j] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            for (long i = firstIdx; i < lastIdx; i++) {
                                long twoi = 2 * i;
                                long idx = offa + i;
                                a.setFloat(idx, wl.getFloat(twoi) * t.getFloat(twoi) - wl.getFloat(twoi + 1) * t.getFloat(twoi + 1));
                            }

                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(FloatDCT_1D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(FloatDCT_1D.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else {
                for (long i = 0; i < nl; i++) {
                    long twoi = 2 * i;
                    long idx = offa + i;
                    a.setFloat(idx, wl.getFloat(twoi) * t.getFloat(twoi) - wl.getFloat(twoi + 1) * t.getFloat(twoi + 1));
                }
            }
            if (scale) {
                CommonUtils.scale(nl, 1 / (float) sqrt(twon), a, offa, false);
                a.setFloat(offa, a.getFloat(offa) / (float) sqrt(2.0));
            }
        }
    }

    /**
     * Computes 1D inverse DCT (DCT-III) leaving the result in <code>a</code>.
     *  
     * @param a
     *              data to transform
     * @param scale
     *              if true then scaling is performed
     */
    public void inverse(float[] a, boolean scale)
    {
        inverse(a, 0, scale);
    }

    /**
     * Computes 1D inverse DCT (DCT-III) leaving the result in <code>a</code>.
     *  
     * @param a
     *              data to transform
     * @param scale
     *              if true then scaling is performed
     */
    public void inverse(FloatLargeArray a, boolean scale)
    {
        inverse(a, 0, scale);
    }

    /**
     * Computes 1D inverse DCT (DCT-III) leaving the result in <code>a</code>.
     *  
     * @param a
     *              data to transform
     * @param offa
     *              index of the first element in array <code>a</code>
     * @param scale
     *              if true then scaling is performed
     */
    public void inverse(final float[] a, final int offa, boolean scale)
    {
        if (n == 1)
            return;
        if (useLargeArrays) {
            inverse(new FloatLargeArray(a), offa, scale);
        } else if (isPowerOfTwo) {
            float xr;
            if (scale) {
                CommonUtils.scale(n, (float) sqrt(2.0 / n), a, offa, false);
                a[offa] = a[offa] / (float) sqrt(2.0);
            }
            CommonUtils.dctsub(n, a, offa, nc, w, nw);
            if (n > 4) {
                CommonUtils.cftfsub(n, a, offa, ip, nw, w);
                rftfsub(n, a, offa, nc, w, nw);
            } else if (n == 4) {
                CommonUtils.cftfsub(n, a, offa, ip, nw, w);
            }
            xr = a[offa] - a[offa + 1];
            a[offa] += a[offa + 1];
            for (int j = 2; j < n; j += 2) {
                a[offa + j - 1] = a[offa + j] - a[offa + j + 1];
                a[offa + j] += a[offa + j + 1];
            }
            a[offa + n - 1] = xr;
        } else {
            final int twon = 2 * n;
            if (scale) {
                CommonUtils.scale(n, (float) sqrt(twon), a, offa, false);
                a[offa] = a[offa] * (float) sqrt(2.0);
            }
            final float[] t = new float[twon];
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && (n > CommonUtils.getThreadsBeginN_1D_FFT_2Threads())) {
                nthreads = 2;
                final int k = n / nthreads;
                Future<?>[] futures = new Future[nthreads];
                for (int j = 0; j < nthreads; j++) {
                    final int firstIdx = j * k;
                    final int lastIdx = (j == (nthreads - 1)) ? n : firstIdx + k;
                    futures[j] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            for (int i = firstIdx; i < lastIdx; i++) {
                                int twoi = 2 * i;
                                float elem = a[offa + i];
                                t[twoi] = w[twoi] * elem;
                                t[twoi + 1] = -w[twoi + 1] * elem;
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(FloatDCT_1D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(FloatDCT_1D.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else {
                for (int i = 0; i < n; i++) {
                    int twoi = 2 * i;
                    float elem = a[offa + i];
                    t[twoi] = w[twoi] * elem;
                    t[twoi + 1] = -w[twoi + 1] * elem;
                }
            }
            fft.realInverse(t, true);
            System.arraycopy(t, 0, a, offa, n);
        }
    }

    /**
     * Computes 1D inverse DCT (DCT-III) leaving the result in <code>a</code>.
     *  
     * @param a
     *              data to transform
     * @param offa
     *              index of the first element in array <code>a</code>
     * @param scale
     *              if true then scaling is performed
     */
    public void inverse(final FloatLargeArray a, final long offa, boolean scale)
    {
        if (nl == 1)
            return;
        if (!useLargeArrays) {
            if (!a.isLarge() && !a.isConstant() && offa < Integer.MAX_VALUE) {
                inverse(a.getData(), (int) offa, scale);
            } else {
                throw new IllegalArgumentException("The data array is too big.");
            }
        } else if (isPowerOfTwo) {
            float xr;
            if (scale) {
                CommonUtils.scale(nl, (float) sqrt(2.0 / nl), a, offa, false);
                a.setFloat(offa, a.getFloat(offa) / (float) sqrt(2.0));
            }
            CommonUtils.dctsub(nl, a, offa, ncl, wl, nwl);
            if (nl > 4) {
                CommonUtils.cftfsub(nl, a, offa, ipl, nwl, wl);
                rftfsub(nl, a, offa, ncl, wl, nwl);
            } else if (nl == 4) {
                CommonUtils.cftfsub(nl, a, offa, ipl, nwl, wl);
            }
            xr = a.getFloat(offa) - a.getFloat(offa + 1);
            a.setFloat(offa, a.getFloat(offa) + a.getFloat(offa + 1));
            for (long j = 2; j < nl; j += 2) {
                a.setFloat(offa + j - 1, a.getFloat(offa + j) - a.getFloat(offa + j + 1));
                a.setFloat(offa + j, a.getFloat(offa + j) + a.getFloat(offa + j + 1));
            }
            a.setFloat(offa + nl - 1, xr);
        } else {
            final long twon = 2 * nl;
            if (scale) {
                CommonUtils.scale(nl, (float) sqrt(twon), a, offa, false);
                a.setFloat(offa, a.getFloat(offa) * (float) sqrt(2.0));
            }
            final FloatLargeArray t = new FloatLargeArray(twon);
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && (nl > CommonUtils.getThreadsBeginN_1D_FFT_2Threads())) {
                nthreads = 2;
                final long k = nl / nthreads;
                Future<?>[] futures = new Future[nthreads];
                for (int j = 0; j < nthreads; j++) {
                    final long firstIdx = j * k;
                    final long lastIdx = (j == (nthreads - 1)) ? nl : firstIdx + k;
                    futures[j] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            for (long i = firstIdx; i < lastIdx; i++) {
                                long twoi = 2 * i;
                                float elem = a.getFloat(offa + i);
                                t.setFloat(twoi, wl.getFloat(twoi) * elem);
                                t.setFloat(twoi + 1, -wl.getFloat(twoi + 1) * elem);
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(FloatDCT_1D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(FloatDCT_1D.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else {
                for (long i = 0; i < nl; i++) {
                    long twoi = 2 * i;
                    float elem = a.getFloat(offa + i);
                    t.setFloat(twoi, wl.getFloat(twoi) * elem);
                    t.setFloat(twoi + 1, -wl.getFloat(twoi + 1) * elem);
                }
            }
            fft.realInverse(t, true);
            LargeArrayUtils.arraycopy(t, 0, a, offa, nl);
        }
    }

    /* -------- initializing routines -------- */
    private float[] makect(int n)
    {
        int twon = 2 * n;
        int idx;
        float delta = PI / twon;
        float deltaj;
        float[] c = new float[twon];
        c[0] = 1;
        for (int j = 1; j < n; j++) {
            idx = 2 * j;
            deltaj = delta * j;
            c[idx] = (float) cos(deltaj);
            c[idx + 1] = -(float) sin(deltaj);
        }
        return c;
    }

    private FloatLargeArray makect(long n)
    {
        long twon = 2 * n;
        long idx;
        float delta = PI / twon;
        float deltaj;
        FloatLargeArray c = new FloatLargeArray(twon);
        c.setFloat(0, 1);
        for (long j = 1; j < n; j++) {
            idx = 2 * j;
            deltaj = delta * j;
            c.setFloat(idx, (float) cos(deltaj));
            c.setFloat(idx + 1, -(float) sin(deltaj));
        }
        return c;
    }

    private static void rftfsub(int n, float[] a, int offa, int nc, float[] c, int startc)
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

    private static void rftfsub(long n, FloatLargeArray a, long offa, long nc, FloatLargeArray c, long startc)
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
            a.setFloat(idx1 + 1, a.getFloat(idx1 + 1) - yi);
            a.setFloat(idx2, a.getFloat(idx2) + yr);
            a.setFloat(idx2 + 1, a.getFloat(idx2 + 1) - yi);
        }
    }

    private static void rftbsub(int n, float[] a, int offa, int nc, float[] c, int startc)
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
            yr = wkr * xr + wki * xi;
            yi = wkr * xi - wki * xr;
            a[idx1] -= yr;
            a[idx1 + 1] -= yi;
            a[idx2] += yr;
            a[idx2 + 1] -= yi;
        }
    }

    private static void rftbsub(long n, FloatLargeArray a, long offa, long nc, FloatLargeArray c, long startc)
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
            yr = wkr * xr + wki * xi;
            yi = wkr * xi - wki * xr;
            a.setFloat(idx1, a.getFloat(idx1) - yr);
            a.setFloat(idx1 + 1, a.getFloat(idx1 + 1) - yi);
            a.setFloat(idx2, a.getFloat(idx2) + yr);
            a.setFloat(idx2 + 1, a.getFloat(idx2 + 1) - yi);
        }
    }
}
