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

import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.jtransforms.fft.DoubleFFT_1D;
import org.jtransforms.utils.CommonUtils;
import pl.edu.icm.jlargearrays.ConcurrencyUtils;
import pl.edu.icm.jlargearrays.DoubleLargeArray;
import pl.edu.icm.jlargearrays.LargeArray;
import pl.edu.icm.jlargearrays.LongLargeArray;
import pl.edu.icm.jlargearrays.LargeArrayUtils;
import static org.apache.commons.math3.util.FastMath.*;

/**
 * Computes 1D Discrete Cosine Transform (DCT) of double precision data. The
 * size of data can be an arbitrary number. This is a parallel implementation of
 * split-radix and mixed-radix algorithms optimized for SMP systems. <br>
 * <br>
 * Part of the code is derived from General Purpose FFT Package written by
 * Takuya Ooura (http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html)
 *  
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class DoubleDCT_1D
{

    private int n;

    private long nl;

    private int[] ip;

    private LongLargeArray ipl;

    private double[] w;

    private DoubleLargeArray wl;

    private int nw;

    private long nwl;

    private int nc;

    private long ncl;

    private boolean isPowerOfTwo = false;

    private DoubleFFT_1D fft;

    private static final double PI = 3.14159265358979311599796346854418516;

    private boolean useLargeArrays;

    /**
     * Creates new instance of DoubleDCT_1D.
     *  
     * @param n size of data
     */
    public DoubleDCT_1D(long n)
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
                this.w = new double[this.n * 5 / 4];
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
                fft = new DoubleFFT_1D(2 * n);
            }
        } else if (CommonUtils.isPowerOf2(n)) {
            this.isPowerOfTwo = true;
            this.ipl = new LongLargeArray((long) ceil(2 + (1l << (long) (log(n / 2 + 0.5) / log(2)) / 2)));
            this.wl = new DoubleLargeArray(this.nl * 5l / 4l);
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
            fft = new DoubleFFT_1D(2 * n);
        }

    }

    /**
     * Computes 1D forward DCT (DCT-II) leaving the result in <code>a</code>.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void forward(double[] a, boolean scale)
    {
        forward(a, 0, scale);
    }

    /**
     * Computes 1D forward DCT (DCT-II) leaving the result in <code>a</code>.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void forward(DoubleLargeArray a, boolean scale)
    {
        forward(a, 0, scale);
    }

    /**
     * Computes 1D forward DCT (DCT-II) leaving the result in <code>a</code>.
     *  
     * @param a     data to transform
     * @param offa  index of the first element in array <code>a</code>
     * @param scale if true then scaling is performed
     */
    public void forward(final double[] a, final int offa, boolean scale)
    {
        if (n == 1) {
            return;
        }
        if (useLargeArrays) {
            forward(new DoubleLargeArray(a), offa, scale);
        } else if (isPowerOfTwo) {
            double xr = a[offa + n - 1];
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
                CommonUtils.scale(n, sqrt(2.0 / n), a, offa, false);
                a[offa] = a[offa] / sqrt(2.0);
            }
        } else {
            final int twon = 2 * n;
            final double[] t = new double[twon];
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
                    Logger.getLogger(DoubleDCT_1D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleDCT_1D.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else {
                for (int i = 0; i < n; i++) {
                    int twoi = 2 * i;
                    int idx = offa + i;
                    a[idx] = w[twoi] * t[twoi] - w[twoi + 1] * t[twoi + 1];
                }
            }
            if (scale) {
                CommonUtils.scale(n, 1 / sqrt(twon), a, offa, false);
                a[offa] = a[offa] / sqrt(2.0);
            }
        }
    }

    /**
     * Computes 1D forward DCT (DCT-II) leaving the result in <code>a</code>.
     *  
     * @param a     data to transform
     * @param offa  index of the first element in array <code>a</code>
     * @param scale if true then scaling is performed
     */
    public void forward(final DoubleLargeArray a, final long offa, boolean scale)
    {
        if (nl == 1) {
            return;
        }
        if (!useLargeArrays) {
            if (!a.isLarge() && !a.isConstant() && offa < Integer.MAX_VALUE) {
                forward(a.getData(), (int) offa, scale);
            } else {
                throw new IllegalArgumentException("The data array is too big.");
            }
        } else if (isPowerOfTwo) {
            double xr = a.getDouble(offa + nl - 1);
            for (long j = nl - 2; j >= 2; j -= 2) {
                a.setDouble(offa + j + 1, a.getDouble(offa + j) - a.getDouble(offa + j - 1));
                a.setDouble(offa + j, a.getDouble(offa + j) + a.getDouble(offa + j - 1));
            }
            a.setDouble(offa + 1, a.getDouble(offa) - xr);
            a.setDouble(offa, a.getDouble(offa) + xr);
            if (nl > 4) {
                rftbsub(nl, a, offa, ncl, wl, nwl);
                CommonUtils.cftbsub(nl, a, offa, ipl, nwl, wl);
            } else if (nl == 4) {
                CommonUtils.cftbsub(nl, a, offa, ipl, nwl, wl);
            }
            CommonUtils.dctsub(nl, a, offa, ncl, wl, nwl);
            if (scale) {
                CommonUtils.scale(nl, sqrt(2.0 / nl), a, offa, false);
                a.setDouble(offa, a.getDouble(offa) / sqrt(2.0));
            }
        } else {
            final long twon = 2 * nl;
            final DoubleLargeArray t = new DoubleLargeArray(twon);
            LargeArrayUtils.arraycopy(a, offa, t, 0, nl);
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            for (long i = nl; i < twon; i++) {
                t.setDouble(i, t.getDouble(twon - i - 1));
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
                                a.setDouble(idx, wl.getDouble(twoi) * t.getDouble(twoi) - wl.getDouble(twoi + 1) * t.getDouble(twoi + 1));
                            }

                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleDCT_1D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleDCT_1D.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else {
                for (long i = 0; i < nl; i++) {
                    long twoi = 2 * i;
                    long idx = offa + i;
                    a.setDouble(idx, wl.getDouble(twoi) * t.getDouble(twoi) - wl.getDouble(twoi + 1) * t.getDouble(twoi + 1));
                }
            }
            if (scale) {
                CommonUtils.scale(nl, 1 / sqrt(twon), a, offa, false);
                a.setDouble(offa, a.getDouble(offa) / sqrt(2.0));
            }
        }
    }

    /**
     * Computes 1D inverse DCT (DCT-III) leaving the result in <code>a</code>.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void inverse(double[] a, boolean scale)
    {
        inverse(a, 0, scale);
    }

    /**
     * Computes 1D inverse DCT (DCT-III) leaving the result in <code>a</code>.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void inverse(DoubleLargeArray a, boolean scale)
    {
        inverse(a, 0, scale);
    }

    /**
     * Computes 1D inverse DCT (DCT-III) leaving the result in <code>a</code>.
     *  
     * @param a     data to transform
     * @param offa  index of the first element in array <code>a</code>
     * @param scale if true then scaling is performed
     */
    public void inverse(final double[] a, final int offa, boolean scale)
    {
        if (n == 1) {
            return;
        }
        if (useLargeArrays) {
            inverse(new DoubleLargeArray(a), offa, scale);
        } else if (isPowerOfTwo) {
            double xr;
            if (scale) {
                CommonUtils.scale(n, sqrt(2.0 / n), a, offa, false);
                a[offa] = a[offa] / sqrt(2.0);
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
                CommonUtils.scale(n, sqrt(twon), a, offa, false);
                a[offa] = a[offa] * sqrt(2.0);
            }
            final double[] t = new double[twon];
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
                                double elem = a[offa + i];
                                t[twoi] = w[twoi] * elem;
                                t[twoi + 1] = -w[twoi + 1] * elem;
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleDCT_1D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleDCT_1D.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else {
                for (int i = 0; i < n; i++) {
                    int twoi = 2 * i;
                    double elem = a[offa + i];
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
     * @param a     data to transform
     * @param offa  index of the first element in array <code>a</code>
     * @param scale if true then scaling is performed
     */
    public void inverse(final DoubleLargeArray a, final long offa, boolean scale)
    {
        if (nl == 1) {
            return;
        }
        if (!useLargeArrays) {
            if (!a.isLarge() && !a.isConstant() && offa < Integer.MAX_VALUE) {
                inverse(a.getData(), (int) offa, scale);
            } else {
                throw new IllegalArgumentException("The data array is too big.");
            }
        } else if (isPowerOfTwo) {
            double xr;
            if (scale) {
                CommonUtils.scale(nl, sqrt(2.0 / nl), a, offa, false);
                a.setDouble(offa, a.getDouble(offa) / sqrt(2.0));
            }
            CommonUtils.dctsub(nl, a, offa, ncl, wl, nwl);
            if (nl > 4) {
                CommonUtils.cftfsub(nl, a, offa, ipl, nwl, wl);
                rftfsub(nl, a, offa, ncl, wl, nwl);
            } else if (nl == 4) {
                CommonUtils.cftfsub(nl, a, offa, ipl, nwl, wl);
            }
            xr = a.getDouble(offa) - a.getDouble(offa + 1);
            a.setDouble(offa, a.getDouble(offa) + a.getDouble(offa + 1));
            for (long j = 2; j < nl; j += 2) {
                a.setDouble(offa + j - 1, a.getDouble(offa + j) - a.getDouble(offa + j + 1));
                a.setDouble(offa + j, a.getDouble(offa + j) + a.getDouble(offa + j + 1));
            }
            a.setDouble(offa + nl - 1, xr);
        } else {
            final long twon = 2 * nl;
            if (scale) {
                CommonUtils.scale(nl, sqrt(twon), a, offa, false);
                a.setDouble(offa, a.getDouble(offa) * sqrt(2.0));
            }
            final DoubleLargeArray t = new DoubleLargeArray(twon);
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
                                double elem = a.getDouble(offa + i);
                                t.setDouble(twoi, wl.getDouble(twoi) * elem);
                                t.setDouble(twoi + 1, -wl.getDouble(twoi + 1) * elem);
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleDCT_1D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleDCT_1D.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else {
                for (long i = 0; i < nl; i++) {
                    long twoi = 2 * i;
                    double elem = a.getDouble(offa + i);
                    t.setDouble(twoi, wl.getDouble(twoi) * elem);
                    t.setDouble(twoi + 1, -wl.getDouble(twoi + 1) * elem);
                }
            }
            fft.realInverse(t, true);
            LargeArrayUtils.arraycopy(t, 0, a, offa, nl);
        }
    }

    /* -------- initializing routines -------- */
    private double[] makect(int n)
    {
        int twon = 2 * n;
        int idx;
        double delta = PI / twon;
        double deltaj;
        double[] c = new double[twon];
        c[0] = 1;
        for (int j = 1; j < n; j++) {
            idx = 2 * j;
            deltaj = delta * j;
            c[idx] = cos(deltaj);
            c[idx + 1] = -sin(deltaj);
        }
        return c;
    }

    private DoubleLargeArray makect(long n)
    {
        long twon = 2 * n;
        long idx;
        double delta = PI / twon;
        double deltaj;
        DoubleLargeArray c = new DoubleLargeArray(twon);
        c.setDouble(0, 1);
        for (long j = 1; j < n; j++) {
            idx = 2 * j;
            deltaj = delta * j;
            c.setDouble(idx, cos(deltaj));
            c.setDouble(idx + 1, -sin(deltaj));
        }
        return c;
    }

    private static void rftfsub(int n, double[] a, int offa, int nc, double[] c, int startc)
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

    private static void rftfsub(long n, DoubleLargeArray a, long offa, long nc, DoubleLargeArray c, long startc)
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

    private static void rftbsub(int n, double[] a, int offa, int nc, double[] c, int startc)
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
            yr = wkr * xr + wki * xi;
            yi = wkr * xi - wki * xr;
            a[idx1] -= yr;
            a[idx1 + 1] -= yi;
            a[idx2] += yr;
            a[idx2 + 1] -= yi;
        }
    }

    private static void rftbsub(long n, DoubleLargeArray a, long offa, long nc, DoubleLargeArray c, long startc)
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
            yr = wkr * xr + wki * xi;
            yi = wkr * xi - wki * xr;
            a.setDouble(idx1, a.getDouble(idx1) - yr);
            a.setDouble(idx1 + 1, a.getDouble(idx1 + 1) - yi);
            a.setDouble(idx2, a.getDouble(idx2) + yr);
            a.setDouble(idx2 + 1, a.getDouble(idx2 + 1) - yi);
        }
    }
}
