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
import java.util.concurrent.ExecutionException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.jtransforms.utils.CommonUtils;
import pl.edu.icm.jlargearrays.ConcurrencyUtils;
import pl.edu.icm.jlargearrays.DoubleLargeArray;
import pl.edu.icm.jlargearrays.LargeArray;
import static org.apache.commons.math3.util.FastMath.*;

/**
 * Computes 2D Discrete Hartley Transform (DHT) of real, double precision data.
 * The sizes of both dimensions can be arbitrary numbers. This is a parallel
 * implementation optimized for SMP systems.<br>
 * <br>
 * Part of code is derived from General Purpose FFT Package written by Takuya
 * Ooura (http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html)
 *  
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class DoubleDHT_2D
{

    private int rows;

    private int columns;

    private long rowsl;

    private long columnsl;

    private DoubleDHT_1D dhtColumns, dhtRows;

    private boolean isPowerOfTwo = false;

    private boolean useThreads = false;

    /**
     * Creates new instance of DoubleDHT_2D.
     *  
     * @param rows    number of rows
     * @param columns number of columns
     */
    public DoubleDHT_2D(long rows, long columns)
    {
        if (rows <= 1 || columns <= 1) {
            throw new IllegalArgumentException("rows and columns must be greater than 1");
        }
        this.rows = (int) rows;
        this.columns = (int) columns;
        this.rowsl = rows;
        this.columnsl = columns;
        if (rows * columns >= CommonUtils.getThreadsBeginN_2D()) {
            this.useThreads = true;
        }
        if (CommonUtils.isPowerOf2(rows) && CommonUtils.isPowerOf2(columns)) {
            isPowerOfTwo = true;
        }
        CommonUtils.setUseLargeArrays(rows * columns > LargeArray.getMaxSizeOf32bitArray());
        dhtRows = new DoubleDHT_1D(rows);
        if (rows == columns) {
            dhtColumns = dhtRows;
        } else {
            dhtColumns = new DoubleDHT_1D(columns);
        }
    }

    /**
     * Computes 2D real, forward DHT leaving the result in <code>a</code>. The
     * data is stored in 1D array in row-major order.
     *  
     * @param a data to transform
     */
    public void forward(final double[] a)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            if ((nthreads > 1) && useThreads) {
                ddxt2d_subth(-1, a, true);
                ddxt2d0_subth(-1, a, true);
            } else {
                ddxt2d_sub(-1, a, true);
                for (int i = 0; i < rows; i++) {
                    dhtColumns.forward(a, i * columns);
                }
            }
            DoubleDHT_2D.this.yTransform(a);
        } else {
            if ((nthreads > 1) && useThreads && (rows >= nthreads) && (columns >= nthreads)) {
                Future<?>[] futures = new Future[nthreads];
                int p = rows / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final int firstRow = l * p;
                    final int lastRow = (l == (nthreads - 1)) ? rows : firstRow + p;
                    futures[l] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            for (int i = firstRow; i < lastRow; i++) {
                                dhtColumns.forward(a, i * columns);
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
                }
                p = columns / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final int firstColumn = l * p;
                    final int lastColumn = (l == (nthreads - 1)) ? columns : firstColumn + p;
                    futures[l] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            double[] temp = new double[rows];
                            for (int c = firstColumn; c < lastColumn; c++) {
                                for (int r = 0; r < rows; r++) {
                                    temp[r] = a[r * columns + c];
                                }
                                dhtRows.forward(temp);
                                for (int r = 0; r < rows; r++) {
                                    a[r * columns + c] = temp[r];
                                }
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
                }

            } else {
                for (int i = 0; i < rows; i++) {
                    dhtColumns.forward(a, i * columns);
                }
                double[] temp = new double[rows];
                for (int c = 0; c < columns; c++) {
                    for (int r = 0; r < rows; r++) {
                        temp[r] = a[r * columns + c];
                    }
                    dhtRows.forward(temp);
                    for (int r = 0; r < rows; r++) {
                        a[r * columns + c] = temp[r];
                    }
                }
            }
            yTransform(a);
        }
    }

    /**
     * Computes 2D real, forward DHT leaving the result in <code>a</code>. The
     * data is stored in 1D array in row-major order.
     *  
     * @param a data to transform
     */
    public void forward(final DoubleLargeArray a)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            if ((nthreads > 1) && useThreads) {
                ddxt2d_subth(-1, a, true);
                ddxt2d0_subth(-1, a, true);
            } else {
                ddxt2d_sub(-1, a, true);
                for (long i = 0; i < rowsl; i++) {
                    dhtColumns.forward(a, i * columnsl);
                }
            }
            yTransform(a);
        } else {
            if ((nthreads > 1) && useThreads && (rowsl >= nthreads) && (columnsl >= nthreads)) {
                Future<?>[] futures = new Future[nthreads];
                long p = rowsl / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final long firstRow = l * p;
                    final long lastRow = (l == (nthreads - 1)) ? rowsl : firstRow + p;
                    futures[l] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            for (long i = firstRow; i < lastRow; i++) {
                                dhtColumns.forward(a, i * columnsl);
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
                }
                p = columnsl / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final long firstColumn = l * p;
                    final long lastColumn = (l == (nthreads - 1)) ? columnsl : firstColumn + p;
                    futures[l] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            DoubleLargeArray temp = new DoubleLargeArray(rowsl, false);
                            for (long c = firstColumn; c < lastColumn; c++) {
                                for (long r = 0; r < rowsl; r++) {
                                    temp.setDouble(r, a.getDouble(r * columnsl + c));
                                }
                                dhtRows.forward(temp);
                                for (long r = 0; r < rowsl; r++) {
                                    a.setDouble(r * columnsl + c, temp.getDouble(r));
                                }
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
                }

            } else {
                for (long i = 0; i < rowsl; i++) {
                    dhtColumns.forward(a, i * columnsl);
                }
                DoubleLargeArray temp = new DoubleLargeArray(rowsl, false);
                for (long c = 0; c < columnsl; c++) {
                    for (long r = 0; r < rowsl; r++) {
                        temp.setDouble(r, a.getDouble(r * columnsl + c));
                    }
                    dhtRows.forward(temp);
                    for (long r = 0; r < rowsl; r++) {
                        a.setDouble(r * columnsl + c, temp.getDouble(r));
                    }
                }
            }
            yTransform(a);
        }
    }

    /**
     * Computes 2D real, forward DHT leaving the result in <code>a</code>. The
     * data is stored in 2D array.
     *  
     * @param a data to transform
     */
    public void forward(final double[][] a)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            if ((nthreads > 1) && useThreads) {
                ddxt2d_subth(-1, a, true);
                ddxt2d0_subth(-1, a, true);
            } else {
                ddxt2d_sub(-1, a, true);
                for (int i = 0; i < rows; i++) {
                    dhtColumns.forward(a[i]);
                }
            }
            yTransform(a);
        } else {
            if ((nthreads > 1) && useThreads && (rows >= nthreads) && (columns >= nthreads)) {
                Future<?>[] futures = new Future[nthreads];
                int p = rows / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final int firstRow = l * p;
                    final int lastRow = (l == (nthreads - 1)) ? rows : firstRow + p;
                    futures[l] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            for (int i = firstRow; i < lastRow; i++) {
                                dhtColumns.forward(a[i]);
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
                }

                p = columns / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final int firstColumn = l * p;
                    final int lastColumn = (l == (nthreads - 1)) ? columns : firstColumn + p;
                    futures[l] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            double[] temp = new double[rows];
                            for (int c = firstColumn; c < lastColumn; c++) {
                                for (int r = 0; r < rows; r++) {
                                    temp[r] = a[r][c];
                                }
                                dhtRows.forward(temp);
                                for (int r = 0; r < rows; r++) {
                                    a[r][c] = temp[r];
                                }
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
                }

            } else {
                for (int i = 0; i < rows; i++) {
                    dhtColumns.forward(a[i]);
                }
                double[] temp = new double[rows];
                for (int c = 0; c < columns; c++) {
                    for (int r = 0; r < rows; r++) {
                        temp[r] = a[r][c];
                    }
                    dhtRows.forward(temp);
                    for (int r = 0; r < rows; r++) {
                        a[r][c] = temp[r];
                    }
                }
            }
            yTransform(a);
        }
    }

    /**
     * Computes 2D real, inverse DHT leaving the result in <code>a</code>. The
     * data is stored in 1D array in row-major order.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void inverse(final double[] a, final boolean scale)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            if ((nthreads > 1) && useThreads) {
                ddxt2d_subth(1, a, scale);
                ddxt2d0_subth(1, a, scale);
            } else {
                ddxt2d_sub(1, a, scale);
                for (int i = 0; i < rows; i++) {
                    dhtColumns.inverse(a, i * columns, scale);
                }
            }
            yTransform(a);
        } else {
            if ((nthreads > 1) && useThreads && (rows >= nthreads) && (columns >= nthreads)) {
                Future<?>[] futures = new Future[nthreads];
                int p = rows / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final int firstRow = l * p;
                    final int lastRow = (l == (nthreads - 1)) ? rows : firstRow + p;
                    futures[l] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            for (int i = firstRow; i < lastRow; i++) {
                                dhtColumns.inverse(a, i * columns, scale);
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
                }

                p = columns / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final int firstColumn = l * p;
                    final int lastColumn = (l == (nthreads - 1)) ? columns : firstColumn + p;
                    futures[l] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            double[] temp = new double[rows];
                            for (int c = firstColumn; c < lastColumn; c++) {
                                for (int r = 0; r < rows; r++) {
                                    temp[r] = a[r * columns + c];
                                }
                                dhtRows.inverse(temp, scale);
                                for (int r = 0; r < rows; r++) {
                                    a[r * columns + c] = temp[r];
                                }
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
                }

            } else {
                for (int i = 0; i < rows; i++) {
                    dhtColumns.inverse(a, i * columns, scale);
                }
                double[] temp = new double[rows];
                for (int c = 0; c < columns; c++) {
                    for (int r = 0; r < rows; r++) {
                        temp[r] = a[r * columns + c];
                    }
                    dhtRows.inverse(temp, scale);
                    for (int r = 0; r < rows; r++) {
                        a[r * columns + c] = temp[r];
                    }
                }
            }
            yTransform(a);
        }
    }

    /**
     * Computes 2D real, inverse DHT leaving the result in <code>a</code>. The
     * data is stored in 1D array in row-major order.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void inverse(final DoubleLargeArray a, final boolean scale)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            if ((nthreads > 1) && useThreads) {
                ddxt2d_subth(1, a, scale);
                ddxt2d0_subth(1, a, scale);
            } else {
                ddxt2d_sub(1, a, scale);
                for (long i = 0; i < rowsl; i++) {
                    dhtColumns.inverse(a, i * columnsl, scale);
                }
            }
            yTransform(a);
        } else {
            if ((nthreads > 1) && useThreads && (rowsl >= nthreads) && (columnsl >= nthreads)) {
                Future<?>[] futures = new Future[nthreads];
                long p = rowsl / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final long firstRow = l * p;
                    final long lastRow = (l == (nthreads - 1)) ? rowsl : firstRow + p;
                    futures[l] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            for (long i = firstRow; i < lastRow; i++) {
                                dhtColumns.inverse(a, i * columnsl, scale);
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
                }

                p = columnsl / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final long firstColumn = l * p;
                    final long lastColumn = (l == (nthreads - 1)) ? columnsl : firstColumn + p;
                    futures[l] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            DoubleLargeArray temp = new DoubleLargeArray(rowsl, false);
                            for (long c = firstColumn; c < lastColumn; c++) {
                                for (long r = 0; r < rowsl; r++) {
                                    temp.setDouble(r, a.getDouble(r * columnsl + c));
                                }
                                dhtRows.inverse(temp, scale);
                                for (long r = 0; r < rowsl; r++) {
                                    a.setDouble(r * columnsl + c, temp.getDouble(r));
                                }
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
                }

            } else {
                for (long i = 0; i < rowsl; i++) {
                    dhtColumns.inverse(a, i * columnsl, scale);
                }
                DoubleLargeArray temp = new DoubleLargeArray(rowsl, false);
                for (long c = 0; c < columnsl; c++) {
                    for (long r = 0; r < rowsl; r++) {
                        temp.setDouble(r, a.getDouble(r * columnsl + c));
                    }
                    dhtRows.inverse(temp, scale);
                    for (long r = 0; r < rowsl; r++) {
                        a.setDouble(r * columnsl + c, temp.getDouble(r));
                    }
                }
            }
            yTransform(a);
        }
    }

    /**
     * Computes 2D real, inverse DHT leaving the result in <code>a</code>. The
     * data is stored in 2D array.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void inverse(final double[][] a, final boolean scale)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            if ((nthreads > 1) && useThreads) {
                ddxt2d_subth(1, a, scale);
                ddxt2d0_subth(1, a, scale);
            } else {
                ddxt2d_sub(1, a, scale);
                for (int i = 0; i < rows; i++) {
                    dhtColumns.inverse(a[i], scale);
                }
            }
            yTransform(a);
        } else {
            if ((nthreads > 1) && useThreads && (rows >= nthreads) && (columns >= nthreads)) {
                Future<?>[] futures = new Future[nthreads];
                int p = rows / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final int firstRow = l * p;
                    final int lastRow = (l == (nthreads - 1)) ? rows : firstRow + p;
                    futures[l] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            for (int i = firstRow; i < lastRow; i++) {
                                dhtColumns.inverse(a[i], scale);
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
                }

                p = columns / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final int firstColumn = l * p;
                    final int lastColumn = (l == (nthreads - 1)) ? columns : firstColumn + p;
                    futures[l] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            double[] temp = new double[rows];
                            for (int c = firstColumn; c < lastColumn; c++) {
                                for (int r = 0; r < rows; r++) {
                                    temp[r] = a[r][c];
                                }
                                dhtRows.inverse(temp, scale);
                                for (int r = 0; r < rows; r++) {
                                    a[r][c] = temp[r];
                                }
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
                }

            } else {
                for (int i = 0; i < rows; i++) {
                    dhtColumns.inverse(a[i], scale);
                }
                double[] temp = new double[rows];
                for (int c = 0; c < columns; c++) {
                    for (int r = 0; r < rows; r++) {
                        temp[r] = a[r][c];
                    }
                    dhtRows.inverse(temp, scale);
                    for (int r = 0; r < rows; r++) {
                        a[r][c] = temp[r];
                    }
                }
            }
            yTransform(a);
        }
    }

    private void ddxt2d_subth(final int isgn, final double[] a, final boolean scale)
    {
        int nthread = min(columns, ConcurrencyUtils.getNumberOfThreads());
        int nt = 4 * rows;
        if (columns == 2) {
            nt >>= 1;
        } else if (columns < 2) {
            nt >>= 2;
        }
        final int ntf = nt;
        final int nthreads = nthread;
        Future<?>[] futures = new Future[nthreads];

        for (int i = 0; i < nthreads; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {
                public void run()
                {
                    int idx1, idx2;
                    double[] t = new double[ntf];
                    if (columns > 2) {
                        if (isgn == -1) {
                            for (int c = 4 * n0; c < columns; c += 4 * nthreads) {
                                for (int r = 0; r < rows; r++) {
                                    idx1 = r * columns + c;
                                    idx2 = rows + r;
                                    t[r] = a[idx1];
                                    t[idx2] = a[idx1 + 1];
                                    t[idx2 + rows] = a[idx1 + 2];
                                    t[idx2 + 2 * rows] = a[idx1 + 3];
                                }
                                dhtRows.forward(t, 0);
                                dhtRows.forward(t, rows);
                                dhtRows.forward(t, 2 * rows);
                                dhtRows.forward(t, 3 * rows);
                                for (int r = 0; r < rows; r++) {
                                    idx1 = r * columns + c;
                                    idx2 = rows + r;
                                    a[idx1] = t[r];
                                    a[idx1 + 1] = t[idx2];
                                    a[idx1 + 2] = t[idx2 + rows];
                                    a[idx1 + 3] = t[idx2 + 2 * rows];
                                }
                            }
                        } else {
                            for (int c = 4 * n0; c < columns; c += 4 * nthreads) {
                                for (int r = 0; r < rows; r++) {
                                    idx1 = r * columns + c;
                                    idx2 = rows + r;
                                    t[r] = a[idx1];
                                    t[idx2] = a[idx1 + 1];
                                    t[idx2 + rows] = a[idx1 + 2];
                                    t[idx2 + 2 * rows] = a[idx1 + 3];
                                }
                                dhtRows.inverse(t, 0, scale);
                                dhtRows.inverse(t, rows, scale);
                                dhtRows.inverse(t, 2 * rows, scale);
                                dhtRows.inverse(t, 3 * rows, scale);
                                for (int r = 0; r < rows; r++) {
                                    idx1 = r * columns + c;
                                    idx2 = rows + r;
                                    a[idx1] = t[r];
                                    a[idx1 + 1] = t[idx2];
                                    a[idx1 + 2] = t[idx2 + rows];
                                    a[idx1 + 3] = t[idx2 + 2 * rows];
                                }
                            }
                        }
                    } else if (columns == 2) {
                        for (int r = 0; r < rows; r++) {
                            idx1 = r * columns + 2 * n0;
                            idx2 = r;
                            t[idx2] = a[idx1];
                            t[idx2 + rows] = a[idx1 + 1];
                        }
                        if (isgn == -1) {
                            dhtRows.forward(t, 0);
                            dhtRows.forward(t, rows);
                        } else {
                            dhtRows.inverse(t, 0, scale);
                            dhtRows.inverse(t, rows, scale);
                        }
                        for (int r = 0; r < rows; r++) {
                            idx1 = r * columns + 2 * n0;
                            idx2 = r;
                            a[idx1] = t[idx2];
                            a[idx1 + 1] = t[idx2 + rows];
                        }
                    }
                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void ddxt2d_subth(final int isgn, final DoubleLargeArray a, final boolean scale)
    {
        int nthread = (int) min(columnsl, ConcurrencyUtils.getNumberOfThreads());
        long nt = 4 * rowsl;
        if (columnsl == 2) {
            nt >>= 1;
        } else if (columnsl < 2) {
            nt >>= 2;
        }
        final long ntf = nt;
        final int nthreads = nthread;
        Future<?>[] futures = new Future[nthreads];

        for (int i = 0; i < nthreads; i++) {
            final long n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {
                public void run()
                {
                    long idx1, idx2;
                    DoubleLargeArray t = new DoubleLargeArray(ntf);
                    if (columnsl > 2) {
                        if (isgn == -1) {
                            for (long c = 4 * n0; c < columnsl; c += 4 * nthreads) {
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = r * columnsl + c;
                                    idx2 = rowsl + r;
                                    t.setDouble(r, a.getDouble(idx1));
                                    t.setDouble(idx2, a.getDouble(idx1 + 1));
                                    t.setDouble(idx2 + rowsl, a.getDouble(idx1 + 2));
                                    t.setDouble(idx2 + 2 * rowsl, a.getDouble(idx1 + 3));
                                }
                                dhtRows.forward(t, 0);
                                dhtRows.forward(t, rowsl);
                                dhtRows.forward(t, 2 * rowsl);
                                dhtRows.forward(t, 3 * rowsl);
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = r * columnsl + c;
                                    idx2 = rowsl + r;
                                    a.setDouble(idx1, t.getDouble(r));
                                    a.setDouble(idx1 + 1, t.getDouble(idx2));
                                    a.setDouble(idx1 + 2, t.getDouble(idx2 + rowsl));
                                    a.setDouble(idx1 + 3, t.getDouble(idx2 + 2 * rowsl));
                                }
                            }
                        } else {
                            for (long c = 4 * n0; c < columnsl; c += 4 * nthreads) {
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = r * columnsl + c;
                                    idx2 = rowsl + r;
                                    t.setDouble(r, a.getDouble(idx1));
                                    t.setDouble(idx2, a.getDouble(idx1 + 1));
                                    t.setDouble(idx2 + rowsl, a.getDouble(idx1 + 2));
                                    t.setDouble(idx2 + 2 * rowsl, a.getDouble(idx1 + 3));
                                }
                                dhtRows.inverse(t, 0, scale);
                                dhtRows.inverse(t, rowsl, scale);
                                dhtRows.inverse(t, 2 * rowsl, scale);
                                dhtRows.inverse(t, 3 * rowsl, scale);
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = r * columnsl + c;
                                    idx2 = rowsl + r;
                                    a.setDouble(idx1, t.getDouble(r));
                                    a.setDouble(idx1 + 1, t.getDouble(idx2));
                                    a.setDouble(idx1 + 2, t.getDouble(idx2 + rowsl));
                                    a.setDouble(idx1 + 3, t.getDouble(idx2 + 2 * rowsl));
                                }
                            }
                        }
                    } else if (columnsl == 2) {
                        for (long r = 0; r < rowsl; r++) {
                            idx1 = r * columnsl + 2 * n0;
                            idx2 = r;
                            t.setDouble(idx2, a.getDouble(idx1));
                            t.setDouble(idx2 + rowsl, a.getDouble(idx1 + 1));
                        }
                        if (isgn == -1) {
                            dhtRows.forward(t, 0);
                            dhtRows.forward(t, rowsl);
                        } else {
                            dhtRows.inverse(t, 0, scale);
                            dhtRows.inverse(t, rowsl, scale);
                        }
                        for (long r = 0; r < rowsl; r++) {
                            idx1 = r * columnsl + 2 * n0;
                            idx2 = r;
                            a.setDouble(idx1, t.getDouble(idx2));
                            a.setDouble(idx1 + 1, t.getDouble(idx2 + rowsl));
                        }
                    }
                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void ddxt2d_subth(final int isgn, final double[][] a, final boolean scale)
    {
        int nthread = min(columns, ConcurrencyUtils.getNumberOfThreads());
        int nt = 4 * rows;
        if (columns == 2) {
            nt >>= 1;
        } else if (columns < 2) {
            nt >>= 2;
        }
        final int ntf = nt;
        final int nthreads = nthread;
        Future<?>[] futures = new Future[nthreads];

        for (int i = 0; i < nthreads; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {
                public void run()
                {
                    int idx2;
                    double[] t = new double[ntf];
                    if (columns > 2) {
                        if (isgn == -1) {
                            for (int c = 4 * n0; c < columns; c += 4 * nthreads) {
                                for (int r = 0; r < rows; r++) {
                                    idx2 = rows + r;
                                    t[r] = a[r][c];
                                    t[idx2] = a[r][c + 1];
                                    t[idx2 + rows] = a[r][c + 2];
                                    t[idx2 + 2 * rows] = a[r][c + 3];
                                }
                                dhtRows.forward(t, 0);
                                dhtRows.forward(t, rows);
                                dhtRows.forward(t, 2 * rows);
                                dhtRows.forward(t, 3 * rows);
                                for (int r = 0; r < rows; r++) {
                                    idx2 = rows + r;
                                    a[r][c] = t[r];
                                    a[r][c + 1] = t[idx2];
                                    a[r][c + 2] = t[idx2 + rows];
                                    a[r][c + 3] = t[idx2 + 2 * rows];
                                }
                            }
                        } else {
                            for (int c = 4 * n0; c < columns; c += 4 * nthreads) {
                                for (int r = 0; r < rows; r++) {
                                    idx2 = rows + r;
                                    t[r] = a[r][c];
                                    t[idx2] = a[r][c + 1];
                                    t[idx2 + rows] = a[r][c + 2];
                                    t[idx2 + 2 * rows] = a[r][c + 3];
                                }
                                dhtRows.inverse(t, 0, scale);
                                dhtRows.inverse(t, rows, scale);
                                dhtRows.inverse(t, 2 * rows, scale);
                                dhtRows.inverse(t, 3 * rows, scale);
                                for (int r = 0; r < rows; r++) {
                                    idx2 = rows + r;
                                    a[r][c] = t[r];
                                    a[r][c + 1] = t[idx2];
                                    a[r][c + 2] = t[idx2 + rows];
                                    a[r][c + 3] = t[idx2 + 2 * rows];
                                }
                            }
                        }
                    } else if (columns == 2) {
                        for (int r = 0; r < rows; r++) {
                            idx2 = r;
                            t[idx2] = a[r][2 * n0];
                            t[idx2 + rows] = a[r][2 * n0 + 1];
                        }
                        if (isgn == -1) {
                            dhtRows.forward(t, 0);
                            dhtRows.forward(t, rows);
                        } else {
                            dhtRows.inverse(t, 0, scale);
                            dhtRows.inverse(t, rows, scale);
                        }
                        for (int r = 0; r < rows; r++) {
                            idx2 = r;
                            a[r][2 * n0] = t[idx2];
                            a[r][2 * n0 + 1] = t[idx2 + rows];
                        }
                    }
                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void ddxt2d0_subth(final int isgn, final double[] a, final boolean scale)
    {
        final int nthreads = ConcurrencyUtils.getNumberOfThreads() > rows ? rows : ConcurrencyUtils.getNumberOfThreads();

        Future<?>[] futures = new Future[nthreads];

        for (int i = 0; i < nthreads; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {

                public void run()
                {
                    if (isgn == -1) {
                        for (int r = n0; r < rows; r += nthreads) {
                            dhtColumns.forward(a, r * columns);
                        }
                    } else {
                        for (int r = n0; r < rows; r += nthreads) {
                            dhtColumns.inverse(a, r * columns, scale);
                        }
                    }
                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void ddxt2d0_subth(final int isgn, final DoubleLargeArray a, final boolean scale)
    {
        final int nthreads = (int) (ConcurrencyUtils.getNumberOfThreads() > rowsl ? rowsl : ConcurrencyUtils.getNumberOfThreads());

        Future<?>[] futures = new Future[nthreads];

        for (int i = 0; i < nthreads; i++) {
            final long n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {

                public void run()
                {
                    if (isgn == -1) {
                        for (long r = n0; r < rowsl; r += nthreads) {
                            dhtColumns.forward(a, r * columnsl);
                        }
                    } else {
                        for (long r = n0; r < rowsl; r += nthreads) {
                            dhtColumns.inverse(a, r * columnsl, scale);
                        }
                    }
                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void ddxt2d0_subth(final int isgn, final double[][] a, final boolean scale)
    {
        final int nthreads = ConcurrencyUtils.getNumberOfThreads() > rows ? rows : ConcurrencyUtils.getNumberOfThreads();

        Future<?>[] futures = new Future[nthreads];

        for (int i = 0; i < nthreads; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {

                public void run()
                {
                    if (isgn == -1) {
                        for (int r = n0; r < rows; r += nthreads) {
                            dhtColumns.forward(a[r]);
                        }
                    } else {
                        for (int r = n0; r < rows; r += nthreads) {
                            dhtColumns.inverse(a[r], scale);
                        }
                    }
                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(DoubleDHT_2D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void ddxt2d_sub(int isgn, double[] a, boolean scale)
    {
        int idx1, idx2;
        int nt = 4 * rows;
        if (columns == 2) {
            nt >>= 1;
        } else if (columns < 2) {
            nt >>= 2;
        }
        double[] t = new double[nt];
        if (columns > 2) {
            if (isgn == -1) {
                for (int c = 0; c < columns; c += 4) {
                    for (int r = 0; r < rows; r++) {
                        idx1 = r * columns + c;
                        idx2 = rows + r;
                        t[r] = a[idx1];
                        t[idx2] = a[idx1 + 1];
                        t[idx2 + rows] = a[idx1 + 2];
                        t[idx2 + 2 * rows] = a[idx1 + 3];
                    }
                    dhtRows.forward(t, 0);
                    dhtRows.forward(t, rows);
                    dhtRows.forward(t, 2 * rows);
                    dhtRows.forward(t, 3 * rows);
                    for (int r = 0; r < rows; r++) {
                        idx1 = r * columns + c;
                        idx2 = rows + r;
                        a[idx1] = t[r];
                        a[idx1 + 1] = t[idx2];
                        a[idx1 + 2] = t[idx2 + rows];
                        a[idx1 + 3] = t[idx2 + 2 * rows];
                    }
                }
            } else {
                for (int c = 0; c < columns; c += 4) {
                    for (int r = 0; r < rows; r++) {
                        idx1 = r * columns + c;
                        idx2 = rows + r;
                        t[r] = a[idx1];
                        t[idx2] = a[idx1 + 1];
                        t[idx2 + rows] = a[idx1 + 2];
                        t[idx2 + 2 * rows] = a[idx1 + 3];
                    }
                    dhtRows.inverse(t, 0, scale);
                    dhtRows.inverse(t, rows, scale);
                    dhtRows.inverse(t, 2 * rows, scale);
                    dhtRows.inverse(t, 3 * rows, scale);
                    for (int r = 0; r < rows; r++) {
                        idx1 = r * columns + c;
                        idx2 = rows + r;
                        a[idx1] = t[r];
                        a[idx1 + 1] = t[idx2];
                        a[idx1 + 2] = t[idx2 + rows];
                        a[idx1 + 3] = t[idx2 + 2 * rows];
                    }
                }
            }
        } else if (columns == 2) {
            for (int r = 0; r < rows; r++) {
                idx1 = r * columns;
                t[r] = a[idx1];
                t[rows + r] = a[idx1 + 1];
            }
            if (isgn == -1) {
                dhtRows.forward(t, 0);
                dhtRows.forward(t, rows);
            } else {
                dhtRows.inverse(t, 0, scale);
                dhtRows.inverse(t, rows, scale);
            }
            for (int r = 0; r < rows; r++) {
                idx1 = r * columns;
                a[idx1] = t[r];
                a[idx1 + 1] = t[rows + r];
            }
        }
    }

    private void ddxt2d_sub(int isgn, DoubleLargeArray a, boolean scale)
    {
        long idx1, idx2;
        long nt = 4 * rowsl;
        if (columnsl == 2) {
            nt >>= 1;
        } else if (columnsl < 2) {
            nt >>= 2;
        }
        DoubleLargeArray t = new DoubleLargeArray(nt);
        if (columnsl > 2) {
            if (isgn == -1) {
                for (long c = 0; c < columnsl; c += 4) {
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = r * columnsl + c;
                        idx2 = rowsl + r;
                        t.setDouble(r, a.getDouble(idx1));
                        t.setDouble(idx2, a.getDouble(idx1 + 1));
                        t.setDouble(idx2 + rowsl, a.getDouble(idx1 + 2));
                        t.setDouble(idx2 + 2 * rowsl, a.getDouble(idx1 + 3));
                    }
                    dhtRows.forward(t, 0);
                    dhtRows.forward(t, rowsl);
                    dhtRows.forward(t, 2 * rowsl);
                    dhtRows.forward(t, 3 * rowsl);
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = r * columnsl + c;
                        idx2 = rowsl + r;
                        a.setDouble(idx1, t.getDouble(r));
                        a.setDouble(idx1 + 1, t.getDouble(idx2));
                        a.setDouble(idx1 + 2, t.getDouble(idx2 + rowsl));
                        a.setDouble(idx1 + 3, t.getDouble(idx2 + 2 * rowsl));
                    }
                }
            } else {
                for (long c = 0; c < columnsl; c += 4) {
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = r * columnsl + c;
                        idx2 = rowsl + r;
                        t.setDouble(r, a.getDouble(idx1));
                        t.setDouble(idx2, a.getDouble(idx1 + 1));
                        t.setDouble(idx2 + rowsl, a.getDouble(idx1 + 2));
                        t.setDouble(idx2 + 2 * rowsl, a.getDouble(idx1 + 3));
                    }
                    dhtRows.inverse(t, 0, scale);
                    dhtRows.inverse(t, rowsl, scale);
                    dhtRows.inverse(t, 2 * rowsl, scale);
                    dhtRows.inverse(t, 3 * rowsl, scale);
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = r * columnsl + c;
                        idx2 = rowsl + r;
                        a.setDouble(idx1, t.getDouble(r));
                        a.setDouble(idx1 + 1, t.getDouble(idx2));
                        a.setDouble(idx1 + 2, t.getDouble(idx2 + rowsl));
                        a.setDouble(idx1 + 3, t.getDouble(idx2 + 2 * rowsl));
                    }
                }
            }
        } else if (columnsl == 2) {
            for (long r = 0; r < rowsl; r++) {
                idx1 = r * columnsl;
                t.setDouble(r, a.getDouble(idx1));
                t.setDouble(rowsl + r, a.getDouble(idx1 + 1));
            }
            if (isgn == -1) {
                dhtRows.forward(t, 0);
                dhtRows.forward(t, rowsl);
            } else {
                dhtRows.inverse(t, 0, scale);
                dhtRows.inverse(t, rowsl, scale);
            }
            for (long r = 0; r < rowsl; r++) {
                idx1 = r * columnsl;
                a.setDouble(idx1, t.getDouble(r));
                a.setDouble(idx1 + 1, t.getDouble(rowsl + r));
            }
        }
    }

    private void ddxt2d_sub(int isgn, double[][] a, boolean scale)
    {
        int idx2;
        int nt = 4 * rows;
        if (columns == 2) {
            nt >>= 1;
        } else if (columns < 2) {
            nt >>= 2;
        }
        double[] t = new double[nt];
        if (columns > 2) {
            if (isgn == -1) {
                for (int c = 0; c < columns; c += 4) {
                    for (int r = 0; r < rows; r++) {
                        idx2 = rows + r;
                        t[r] = a[r][c];
                        t[idx2] = a[r][c + 1];
                        t[idx2 + rows] = a[r][c + 2];
                        t[idx2 + 2 * rows] = a[r][c + 3];
                    }
                    dhtRows.forward(t, 0);
                    dhtRows.forward(t, rows);
                    dhtRows.forward(t, 2 * rows);
                    dhtRows.forward(t, 3 * rows);
                    for (int r = 0; r < rows; r++) {
                        idx2 = rows + r;
                        a[r][c] = t[r];
                        a[r][c + 1] = t[idx2];
                        a[r][c + 2] = t[idx2 + rows];
                        a[r][c + 3] = t[idx2 + 2 * rows];
                    }
                }
            } else {
                for (int c = 0; c < columns; c += 4) {
                    for (int r = 0; r < rows; r++) {
                        idx2 = rows + r;
                        t[r] = a[r][c];
                        t[idx2] = a[r][c + 1];
                        t[idx2 + rows] = a[r][c + 2];
                        t[idx2 + 2 * rows] = a[r][c + 3];
                    }
                    dhtRows.inverse(t, 0, scale);
                    dhtRows.inverse(t, rows, scale);
                    dhtRows.inverse(t, 2 * rows, scale);
                    dhtRows.inverse(t, 3 * rows, scale);
                    for (int r = 0; r < rows; r++) {
                        idx2 = rows + r;
                        a[r][c] = t[r];
                        a[r][c + 1] = t[idx2];
                        a[r][c + 2] = t[idx2 + rows];
                        a[r][c + 3] = t[idx2 + 2 * rows];
                    }
                }
            }
        } else if (columns == 2) {
            for (int r = 0; r < rows; r++) {
                t[r] = a[r][0];
                t[rows + r] = a[r][1];
            }
            if (isgn == -1) {
                dhtRows.forward(t, 0);
                dhtRows.forward(t, rows);
            } else {
                dhtRows.inverse(t, 0, scale);
                dhtRows.inverse(t, rows, scale);
            }
            for (int r = 0; r < rows; r++) {
                a[r][0] = t[r];
                a[r][1] = t[rows + r];
            }
        }
    }

    private void yTransform(double[] a)
    {
        int mRow, mCol, idx1, idx2;
        double A, B, C, D, E;
        for (int r = 0; r <= rows / 2; r++) {
            mRow = (rows - r) % rows;
            idx1 = r * columns;
            idx2 = mRow * columns;
            for (int c = 0; c <= columns / 2; c++) {
                mCol = (columns - c) % columns;
                A = a[idx1 + c];
                B = a[idx2 + c];
                C = a[idx1 + mCol];
                D = a[idx2 + mCol];
                E = ((A + D) - (B + C)) / 2;
                a[idx1 + c] = A - E;
                a[idx2 + c] = B + E;
                a[idx1 + mCol] = C + E;
                a[idx2 + mCol] = D - E;
            }
        }
    }

    private void yTransform(DoubleLargeArray a)
    {
        long mRow, mCol, idx1, idx2;
        double A, B, C, D, E;
        for (long r = 0; r <= rowsl / 2; r++) {
            mRow = (rowsl - r) % rowsl;
            idx1 = r * columnsl;
            idx2 = mRow * columnsl;
            for (long c = 0; c <= columnsl / 2; c++) {
                mCol = (columnsl - c) % columnsl;
                A = a.getDouble(idx1 + c);
                B = a.getDouble(idx2 + c);
                C = a.getDouble(idx1 + mCol);
                D = a.getDouble(idx2 + mCol);
                E = ((A + D) - (B + C)) / 2;
                a.setDouble(idx1 + c, A - E);
                a.setDouble(idx2 + c, B + E);
                a.setDouble(idx1 + mCol, C + E);
                a.setDouble(idx2 + mCol, D - E);
            }
        }
    }

    private void yTransform(double[][] a)
    {
        int mRow, mCol;
        double A, B, C, D, E;
        for (int r = 0; r <= rows / 2; r++) {
            mRow = (rows - r) % rows;
            for (int c = 0; c <= columns / 2; c++) {
                mCol = (columns - c) % columns;
                A = a[r][c];
                B = a[mRow][c];
                C = a[r][mCol];
                D = a[mRow][mCol];
                E = ((A + D) - (B + C)) / 2;
                a[r][c] = A - E;
                a[mRow][c] = B + E;
                a[r][mCol] = C + E;
                a[mRow][mCol] = D - E;
            }
        }
    }

}
