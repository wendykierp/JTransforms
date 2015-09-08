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

import java.util.concurrent.Future;
import org.jtransforms.fft.DoubleFFT_1D;
import org.jtransforms.utils.CommonUtils;
import java.util.concurrent.ExecutionException;
import java.util.logging.Level;
import java.util.logging.Logger;
import pl.edu.icm.jlargearrays.ConcurrencyUtils;
import pl.edu.icm.jlargearrays.DoubleLargeArray;
import pl.edu.icm.jlargearrays.LargeArray;
import pl.edu.icm.jlargearrays.LargeArrayUtils;

/**
 * Computes 1D Discrete Hartley Transform (DHT) of real, double precision data.
 * The size of the data can be an arbitrary number. It uses FFT algorithm. This
 * is a parallel implementation optimized for SMP systems.
 *  
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class DoubleDHT_1D
{

    private final int n;
    private final long nl;
    private final DoubleFFT_1D fft;
    private final boolean useLargeArrays;

    /**
     * Creates new instance of DoubleDHT_1D.
     *  
     * @param n size of data
     */
    public DoubleDHT_1D(long n)
    {
        this.n = (int) n;
        this.nl = n;
        this.useLargeArrays = (CommonUtils.isUseLargeArrays() || n > LargeArray.getMaxSizeOf32bitArray());
        fft = new DoubleFFT_1D(n);
    }

    /**
     * Computes 1D real, forward DHT leaving the result in <code>a</code>.
     *  
     * @param a data to transform
     */
    public void forward(double[] a)
    {
        forward(a, 0);
    }

    /**
     * Computes 1D real, forward DHT leaving the result in <code>a</code>.
     *  
     * @param a data to transform
     */
    public void forward(DoubleLargeArray a)
    {
        forward(a, 0);
    }

    /**
     * Computes 1D real, forward DHT leaving the result in <code>a</code>.
     *  
     * @param a    data to transform
     * @param offa index of the first element in array <code>a</code>
     */
    public void forward(final double[] a, final int offa)
    {
        if (n == 1) {
            return;
        }
        if (useLargeArrays) {
            forward(new DoubleLargeArray(a), offa);
        } else {
            fft.realForward(a, offa);
            final double[] b = new double[n];
            System.arraycopy(a, offa, b, 0, n);
            int nd2 = n / 2;
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && (nd2 > CommonUtils.getThreadsBeginN_1D_FFT_2Threads())) {
                nthreads = 2;
                final int k1 = nd2 / nthreads;
                Future<?>[] futures = new Future[nthreads];
                for (int i = 0; i < nthreads; i++) {
                    final int firstIdx = 1 + i * k1;
                    final int lastIdx = (i == (nthreads - 1)) ? nd2 : firstIdx + k1;
                    futures[i] = ConcurrencyUtils.submit(new Runnable()
                    {

                        public void run()
                        {
                            int idx1, idx2;
                            for (int i = firstIdx; i < lastIdx; i++) {
                                idx1 = 2 * i;
                                idx2 = idx1 + 1;
                                a[offa + i] = b[idx1] - b[idx2];
                                a[offa + n - i] = b[idx1] + b[idx2];
                            }
                        }

                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleDHT_1D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleDHT_1D.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else {
                int idx1, idx2;
                for (int i = 1; i < nd2; i++) {
                    idx1 = 2 * i;
                    idx2 = idx1 + 1;
                    a[offa + i] = b[idx1] - b[idx2];
                    a[offa + n - i] = b[idx1] + b[idx2];
                }
            }
            if ((n % 2) == 0) {
                a[offa + nd2] = b[1];
            } else {
                a[offa + nd2] = b[n - 1] - b[1];
                a[offa + nd2 + 1] = b[n - 1] + b[1];
            }
        }
    }

    /**
     * Computes 1D real, forward DHT leaving the result in <code>a</code>.
     *  
     * @param a    data to transform
     * @param offa index of the first element in array <code>a</code>
     */
    public void forward(final DoubleLargeArray a, final long offa)
    {
        if (nl == 1) {
            return;
        }
        if (!useLargeArrays) {
            if (!a.isLarge() && !a.isConstant() && offa < Integer.MAX_VALUE) {
                forward(a.getData(), (int) offa);
            } else {
                throw new IllegalArgumentException("The data array is too big.");
            }
        } else {
            fft.realForward(a, offa);
            final DoubleLargeArray b = new DoubleLargeArray(nl, false);
            LargeArrayUtils.arraycopy(a, offa, b, 0, nl);
            long nd2 = nl / 2;
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && (nd2 > CommonUtils.getThreadsBeginN_1D_FFT_2Threads())) {
                nthreads = 2;
                final long k1 = nd2 / nthreads;
                Future<?>[] futures = new Future[nthreads];
                for (int i = 0; i < nthreads; i++) {
                    final long firstIdx = 1 + i * k1;
                    final long lastIdx = (i == (nthreads - 1)) ? nd2 : firstIdx + k1;
                    futures[i] = ConcurrencyUtils.submit(new Runnable()
                    {

                        public void run()
                        {
                            long idx1, idx2;
                            for (long i = firstIdx; i < lastIdx; i++) {
                                idx1 = 2 * i;
                                idx2 = idx1 + 1;
                                a.setDouble(offa + i, b.getDouble(idx1) - b.getDouble(idx2));
                                a.setDouble(offa + nl - i, b.getDouble(idx1) + b.getDouble(idx2));
                            }
                        }

                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleDHT_1D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleDHT_1D.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else {
                long idx1, idx2;
                for (long i = 1; i < nd2; i++) {
                    idx1 = 2 * i;
                    idx2 = idx1 + 1;
                    a.setDouble(offa + i, b.getDouble(idx1) - b.getDouble(idx2));
                    a.setDouble(offa + nl - i, b.getDouble(idx1) + b.getDouble(idx2));
                }
            }
            if ((nl % 2) == 0) {
                a.setDouble(offa + nd2, b.getDouble(1));
            } else {
                a.setDouble(offa + nd2, b.getDouble(nl - 1) - b.getDouble(1));
                a.setDouble(offa + nd2 + 1, b.getDouble(nl - 1) + b.getDouble(1));
            }
        }
    }

    /**
     * Computes 1D real, inverse DHT leaving the result in <code>a</code>.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void inverse(double[] a, boolean scale)
    {
        inverse(a, 0, scale);
    }

    /**
     * Computes 1D real, inverse DHT leaving the result in <code>a</code>.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void inverse(DoubleLargeArray a, boolean scale)
    {
        inverse(a, 0, scale);
    }

    /**
     * Computes 1D real, inverse DHT leaving the result in <code>a</code>.
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
            forward(a, offa);
            if (scale) {
                CommonUtils.scale(n, 1.0 / n, a, offa, false);
            }
        }
    }

    /**
     * Computes 1D real, inverse DHT leaving the result in <code>a</code>.
     *  
     * @param a     data to transform
     * @param offa  index of the first element in array <code>a</code>
     * @param scale if true then scaling is performed
     */
    public void inverse(final DoubleLargeArray a, final long offa, boolean scale)
    {
        if (n == 1) {
            return;
        }
        if (!useLargeArrays) {
            if (!a.isLarge() && !a.isConstant() && offa < Integer.MAX_VALUE) {
                inverse(a.getData(), (int) offa, scale);
            } else {
                throw new IllegalArgumentException("The data array is too big.");
            }
        } else {
            forward(a, offa);
            if (scale) {
                CommonUtils.scale(n, 1.0 / n, a, offa, false);
            }
        }
    }
}
