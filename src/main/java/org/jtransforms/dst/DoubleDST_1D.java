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

import java.util.concurrent.Future;
import org.jtransforms.dct.DoubleDCT_1D;
import java.util.concurrent.ExecutionException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.jtransforms.utils.CommonUtils;
import pl.edu.icm.jlargearrays.ConcurrencyUtils;
import pl.edu.icm.jlargearrays.DoubleLargeArray;
import pl.edu.icm.jlargearrays.LargeArray;

/**
 * Computes 1D Discrete Sine Transform (DST) of double precision data. The size
 * of data can be an arbitrary number. It uses DCT algorithm. This is a parallel
 * implementation optimized for SMP systems.
 *  
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class DoubleDST_1D
{

    private final int n;
    private final long nl;
    private final DoubleDCT_1D dct;
    private final boolean useLargeArrays;

    /**
     * Creates new instance of DoubleDST_1D.
     *  
     * @param n size of data
     */
    public DoubleDST_1D(long n)
    {
        this.n = (int) n;
        this.nl = n;
        this.useLargeArrays = (CommonUtils.isUseLargeArrays() || n > LargeArray.getMaxSizeOf32bitArray());
        dct = new DoubleDCT_1D(n);
    }

    /**
     * Computes 1D forward DST (DST-II) leaving the result in <code>a</code>.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void forward(double[] a, boolean scale)
    {
        forward(a, 0, scale);
    }

    /**
     * Computes 1D forward DST (DST-II) leaving the result in <code>a</code>.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void forward(DoubleLargeArray a, boolean scale)
    {
        forward(a, 0, scale);
    }

    /**
     * Computes 1D forward DST (DST-II) leaving the result in <code>a</code>.
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
        } else {
            double tmp;
            int nd2 = n / 2;
            int startIdx = 1 + offa;
            int stopIdx = offa + n;
            for (int i = startIdx; i < stopIdx; i += 2) {
                a[i] = -a[i];
            }
            dct.forward(a, offa, scale);
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && (nd2 > CommonUtils.getThreadsBeginN_1D_FFT_2Threads())) {
                nthreads = 2;
                final int k = nd2 / nthreads;
                Future<?>[] futures = new Future[nthreads];
                for (int j = 0; j < nthreads; j++) {
                    final int firstIdx = j * k;
                    final int lastIdx = (j == (nthreads - 1)) ? nd2 : firstIdx + k;
                    futures[j] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            double tmp;
                            int idx0 = offa + n - 1;
                            int idx1;
                            int idx2;
                            for (int i = firstIdx; i < lastIdx; i++) {
                                idx2 = offa + i;
                                tmp = a[idx2];
                                idx1 = idx0 - i;
                                a[idx2] = a[idx1];
                                a[idx1] = tmp;
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleDST_1D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleDST_1D.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else {
                int idx0 = offa + n - 1;
                int idx1;
                int idx2;
                for (int i = 0; i < nd2; i++) {
                    idx2 = offa + i;
                    tmp = a[idx2];
                    idx1 = idx0 - i;
                    a[idx2] = a[idx1];
                    a[idx1] = tmp;
                }
            }
        }
    }

    /**
     * Computes 1D forward DST (DST-II) leaving the result in <code>a</code>.
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
        } else {
            double tmp;
            long nd2 = nl / 2;
            long startIdx = 1 + offa;
            long stopIdx = offa + nl;
            for (long i = startIdx; i < stopIdx; i += 2) {
                a.setDouble(i, -a.getDouble(i));
            }
            dct.forward(a, offa, scale);
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && (nd2 > CommonUtils.getThreadsBeginN_1D_FFT_2Threads())) {
                nthreads = 2;
                final long k = nd2 / nthreads;
                Future<?>[] futures = new Future[nthreads];
                for (int j = 0; j < nthreads; j++) {
                    final long firstIdx = j * k;
                    final long lastIdx = (j == (nthreads - 1)) ? nd2 : firstIdx + k;
                    futures[j] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            double tmp;
                            long idx0 = offa + nl - 1;
                            long idx1;
                            long idx2;
                            for (long i = firstIdx; i < lastIdx; i++) {
                                idx2 = offa + i;
                                tmp = a.getDouble(idx2);
                                idx1 = idx0 - i;
                                a.setDouble(idx2, a.getDouble(idx1));
                                a.setDouble(idx1, tmp);
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleDST_1D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleDST_1D.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else {
                long idx0 = offa + nl - 1;
                long idx1;
                long idx2;
                for (long i = 0; i < nd2; i++) {
                    idx2 = offa + i;
                    tmp = a.getDouble(idx2);
                    idx1 = idx0 - i;
                    a.setDouble(idx2, a.getDouble(idx1));
                    a.setDouble(idx1, tmp);
                }
            }
        }
    }

    /**
     * Computes 1D inverse DST (DST-III) leaving the result in <code>a</code>.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void inverse(double[] a, boolean scale)
    {
        inverse(a, 0, scale);
    }

    /**
     * Computes 1D inverse DST (DST-III) leaving the result in <code>a</code>.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void inverse(DoubleLargeArray a, boolean scale)
    {
        inverse(a, 0, scale);
    }

    /**
     * Computes 1D inverse DST (DST-III) leaving the result in <code>a</code>.
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
        } else {
            double tmp;
            int nd2 = n / 2;
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && (nd2 > CommonUtils.getThreadsBeginN_1D_FFT_2Threads())) {
                nthreads = 2;
                final int k = nd2 / nthreads;
                Future<?>[] futures = new Future[nthreads];
                for (int j = 0; j < nthreads; j++) {
                    final int firstIdx = j * k;
                    final int lastIdx = (j == (nthreads - 1)) ? nd2 : firstIdx + k;
                    futures[j] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            double tmp;
                            int idx0 = offa + n - 1;
                            int idx1, idx2;
                            for (int i = firstIdx; i < lastIdx; i++) {
                                idx2 = offa + i;
                                tmp = a[idx2];
                                idx1 = idx0 - i;
                                a[idx2] = a[idx1];
                                a[idx1] = tmp;
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleDST_1D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleDST_1D.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else {
                int idx0 = offa + n - 1;
                for (int i = 0; i < nd2; i++) {
                    tmp = a[offa + i];
                    a[offa + i] = a[idx0 - i];
                    a[idx0 - i] = tmp;
                }
            }
            dct.inverse(a, offa, scale);
            int startidx = 1 + offa;
            int stopidx = offa + n;
            for (int i = startidx; i < stopidx; i += 2) {
                a[i] = -a[i];
            }
        }
    }

    /**
     * Computes 1D inverse DST (DST-III) leaving the result in <code>a</code>.
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
        } else {
            double tmp;
            long nd2 = nl / 2;
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && (nd2 > CommonUtils.getThreadsBeginN_1D_FFT_2Threads())) {
                nthreads = 2;
                final long k = nd2 / nthreads;
                Future<?>[] futures = new Future[nthreads];
                for (int j = 0; j < nthreads; j++) {
                    final long firstIdx = j * k;
                    final long lastIdx = (j == (nthreads - 1)) ? nd2 : firstIdx + k;
                    futures[j] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            double tmp;
                            long idx0 = offa + nl - 1;
                            long idx1, idx2;
                            for (long i = firstIdx; i < lastIdx; i++) {
                                idx2 = offa + i;
                                tmp = a.getDouble(idx2);
                                idx1 = idx0 - i;
                                a.setDouble(idx2, a.getDouble(idx1));
                                a.setDouble(idx1, tmp);
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleDST_1D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleDST_1D.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else {
                long idx0 = offa + nl - 1;
                for (long i = 0; i < nd2; i++) {
                    tmp = a.getDouble(offa + i);
                    a.setDouble(offa + i, a.getDouble(idx0 - i));
                    a.setDouble(idx0 - i, tmp);
                }
            }
            dct.inverse(a, offa, scale);
            long startidx = 1 + offa;
            long stopidx = offa + nl;
            for (long i = startidx; i < stopidx; i += 2) {
                a.setDouble(i, -a.getDouble(i));
            }
        }
    }
}
