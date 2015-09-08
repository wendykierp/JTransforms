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
import java.util.concurrent.ExecutionException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.jtransforms.utils.CommonUtils;
import pl.edu.icm.jlargearrays.ConcurrencyUtils;
import pl.edu.icm.jlargearrays.FloatLargeArray;
import pl.edu.icm.jlargearrays.LargeArray;
import static org.apache.commons.math3.util.FastMath.*;

/**
 * Computes 2D Discrete Cosine Transform (DCT) of single precision data. The
 * sizes of both dimensions can be arbitrary numbers. This is a parallel
 * implementation of split-radix and mixed-radix algorithms optimized for SMP
 * systems. <br>
 * <br>
 * Part of the code is derived from General Purpose FFT Package written by
 * Takuya Ooura (http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html)
 *  
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class FloatDCT_2D
{

    private int rows;

    private int columns;

    private long rowsl;

    private long columnsl;

    private FloatDCT_1D dctColumns, dctRows;

    private boolean isPowerOfTwo = false;

    private boolean useThreads = false;

    /**
     * Creates new instance of FloatDCT_2D.
     *  
     * @param rows    number of rows
     * @param columns number of columns
     */
    public FloatDCT_2D(long rows, long columns)
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
        dctRows = new FloatDCT_1D(rows);
        if (rows == columns) {
            dctColumns = dctRows;
        } else {
            dctColumns = new FloatDCT_1D(columns);
        }
    }

    /**
     * Computes 2D forward DCT (DCT-II) leaving the result in <code>a</code>.
     * The data is stored in 1D array in row-major order.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void forward(final float[] a, final boolean scale)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            if ((nthreads > 1) && useThreads) {
                ddxt2d_subth(-1, a, scale);
                ddxt2d0_subth(-1, a, scale);
            } else {
                ddxt2d_sub(-1, a, scale);
                for (int i = 0; i < rows; i++) {
                    dctColumns.forward(a, i * columns, scale);
                }
            }
        } else if ((nthreads > 1) && useThreads && (rows >= nthreads) && (columns >= nthreads)) {
            Future<?>[] futures = new Future[nthreads];
            int p = rows / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstRow = l * p;
                final int lastRow = (l == (nthreads - 1)) ? rows : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (int r = firstRow; r < lastRow; r++) {
                            dctColumns.forward(a, r * columns, scale);
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }
            p = columns / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstColumn = l * p;
                final int lastColumn = (l == (nthreads - 1)) ? columns : firstColumn + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        float[] temp = new float[rows];
                        for (int c = firstColumn; c < lastColumn; c++) {
                            for (int r = 0; r < rows; r++) {
                                temp[r] = a[r * columns + c];
                            }
                            dctRows.forward(temp, scale);
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
                Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {
            for (int i = 0; i < rows; i++) {
                dctColumns.forward(a, i * columns, scale);
            }
            float[] temp = new float[rows];
            for (int c = 0; c < columns; c++) {
                for (int r = 0; r < rows; r++) {
                    temp[r] = a[r * columns + c];
                }
                dctRows.forward(temp, scale);
                for (int r = 0; r < rows; r++) {
                    a[r * columns + c] = temp[r];
                }
            }
        }
    }

    /**
     * Computes 2D forward DCT (DCT-II) leaving the result in <code>a</code>.
     * The data is stored in 1D array in row-major order.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void forward(final FloatLargeArray a, final boolean scale)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            if ((nthreads > 1) && useThreads) {
                ddxt2d_subth(-1, a, scale);
                ddxt2d0_subth(-1, a, scale);
            } else {
                ddxt2d_sub(-1, a, scale);
                for (long i = 0; i < rowsl; i++) {
                    dctColumns.forward(a, i * columnsl, scale);
                }
            }
        } else if ((nthreads > 1) && useThreads && (rowsl >= nthreads) && (columnsl >= nthreads)) {
            Future<?>[] futures = new Future[nthreads];
            long p = rowsl / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final long firstRow = l * p;
                final long lastRow = (l == (nthreads - 1)) ? rowsl : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (long r = firstRow; r < lastRow; r++) {
                            dctColumns.forward(a, r * columnsl, scale);
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }
            p = columnsl / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final long firstColumn = l * p;
                final long lastColumn = (l == (nthreads - 1)) ? columnsl : firstColumn + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        FloatLargeArray temp = new FloatLargeArray(rowsl, false);
                        for (long c = firstColumn; c < lastColumn; c++) {
                            for (long r = 0; r < rowsl; r++) {
                                temp.setFloat(r, a.getFloat(r * columnsl + c));
                            }
                            dctRows.forward(temp, scale);
                            for (long r = 0; r < rowsl; r++) {
                                a.setFloat(r * columnsl + c, temp.getFloat(r));
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {
            for (long i = 0; i < rowsl; i++) {
                dctColumns.forward(a, i * columnsl, scale);
            }
            FloatLargeArray temp = new FloatLargeArray(rowsl, false);
            for (long c = 0; c < columnsl; c++) {
                for (long r = 0; r < rowsl; r++) {
                    temp.setFloat(r, a.getFloat(r * columnsl + c));
                }
                dctRows.forward(temp, scale);
                for (long r = 0; r < rowsl; r++) {
                    a.setFloat(r * columnsl + c, temp.getFloat(r));
                }
            }
        }
    }

    /**
     * Computes 2D forward DCT (DCT-II) leaving the result in <code>a</code>.
     * The data is stored in 2D array.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void forward(final float[][] a, final boolean scale)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            if ((nthreads > 1) && useThreads) {
                ddxt2d_subth(-1, a, scale);
                ddxt2d0_subth(-1, a, scale);
            } else {
                ddxt2d_sub(-1, a, scale);
                for (int i = 0; i < rows; i++) {
                    dctColumns.forward(a[i], scale);
                }
            }
        } else if ((nthreads > 1) && useThreads && (rows >= nthreads) && (columns >= nthreads)) {
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
                            dctColumns.forward(a[i], scale);
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }
            p = columns / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstColumn = l * p;
                final int lastColumn = (l == (nthreads - 1)) ? columns : firstColumn + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        float[] temp = new float[rows];
                        for (int c = firstColumn; c < lastColumn; c++) {
                            for (int r = 0; r < rows; r++) {
                                temp[r] = a[r][c];
                            }
                            dctRows.forward(temp, scale);
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
                Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {
            for (int i = 0; i < rows; i++) {
                dctColumns.forward(a[i], scale);
            }
            float[] temp = new float[rows];
            for (int c = 0; c < columns; c++) {
                for (int r = 0; r < rows; r++) {
                    temp[r] = a[r][c];
                }
                dctRows.forward(temp, scale);
                for (int r = 0; r < rows; r++) {
                    a[r][c] = temp[r];
                }
            }
        }
    }

    /**
     * Computes 2D inverse DCT (DCT-III) leaving the result in <code>a</code>.
     * The data is stored in 1D array in row-major order.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void inverse(final float[] a, final boolean scale)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            if ((nthreads > 1) && useThreads) {
                ddxt2d_subth(1, a, scale);
                ddxt2d0_subth(1, a, scale);
            } else {
                ddxt2d_sub(1, a, scale);
                for (int i = 0; i < rows; i++) {
                    dctColumns.inverse(a, i * columns, scale);
                }
            }
        } else if ((nthreads > 1) && useThreads && (rows >= nthreads) && (columns >= nthreads)) {
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
                            dctColumns.inverse(a, i * columns, scale);
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }
            p = columns / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstColumn = l * p;
                final int lastColumn = (l == (nthreads - 1)) ? columns : firstColumn + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        float[] temp = new float[rows];
                        for (int c = firstColumn; c < lastColumn; c++) {
                            for (int r = 0; r < rows; r++) {
                                temp[r] = a[r * columns + c];
                            }
                            dctRows.inverse(temp, scale);
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
                Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {
            for (int i = 0; i < rows; i++) {
                dctColumns.inverse(a, i * columns, scale);
            }
            float[] temp = new float[rows];
            for (int c = 0; c < columns; c++) {
                for (int r = 0; r < rows; r++) {
                    temp[r] = a[r * columns + c];
                }
                dctRows.inverse(temp, scale);
                for (int r = 0; r < rows; r++) {
                    a[r * columns + c] = temp[r];
                }
            }
        }
    }

    /**
     * Computes 2D inverse DCT (DCT-III) leaving the result in <code>a</code>.
     * The data is stored in 1D array in row-major order.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void inverse(final FloatLargeArray a, final boolean scale)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            if ((nthreads > 1) && useThreads) {
                ddxt2d_subth(1, a, scale);
                ddxt2d0_subth(1, a, scale);
            } else {
                ddxt2d_sub(1, a, scale);
                for (long i = 0; i < rowsl; i++) {
                    dctColumns.inverse(a, i * columnsl, scale);
                }
            }
        } else if ((nthreads > 1) && useThreads && (rowsl >= nthreads) && (columnsl >= nthreads)) {
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
                            dctColumns.inverse(a, i * columnsl, scale);
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }
            p = columnsl / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final long firstColumn = l * p;
                final long lastColumn = (l == (nthreads - 1)) ? columnsl : firstColumn + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        FloatLargeArray temp = new FloatLargeArray(rowsl, false);
                        for (long c = firstColumn; c < lastColumn; c++) {
                            for (long r = 0; r < rowsl; r++) {
                                temp.setFloat(r, a.getFloat(r * columnsl + c));
                            }
                            dctRows.inverse(temp, scale);
                            for (long r = 0; r < rowsl; r++) {
                                a.setFloat(r * columnsl + c, temp.getFloat(r));
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {
            for (long i = 0; i < rowsl; i++) {
                dctColumns.inverse(a, i * columnsl, scale);
            }
            FloatLargeArray temp = new FloatLargeArray(rowsl, false);
            for (long c = 0; c < columnsl; c++) {
                for (long r = 0; r < rowsl; r++) {
                    temp.setFloat(r, a.getFloat(r * columnsl + c));
                }
                dctRows.inverse(temp, scale);
                for (long r = 0; r < rowsl; r++) {
                    a.setFloat(r * columnsl + c, temp.getFloat(r));
                }
            }
        }
    }

    /**
     * Computes 2D inverse DCT (DCT-III) leaving the result in <code>a</code>.
     * The data is stored in 2D array.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void inverse(final float[][] a, final boolean scale)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            if ((nthreads > 1) && useThreads) {
                ddxt2d_subth(1, a, scale);
                ddxt2d0_subth(1, a, scale);
            } else {
                ddxt2d_sub(1, a, scale);
                for (int i = 0; i < rows; i++) {
                    dctColumns.inverse(a[i], scale);
                }
            }
        } else if ((nthreads > 1) && useThreads && (rows >= nthreads) && (columns >= nthreads)) {
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
                            dctColumns.inverse(a[i], scale);
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }
            p = columns / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstColumn = l * p;
                final int lastColumn = (l == (nthreads - 1)) ? columns : firstColumn + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        float[] temp = new float[rows];
                        for (int c = firstColumn; c < lastColumn; c++) {
                            for (int r = 0; r < rows; r++) {
                                temp[r] = a[r][c];
                            }
                            dctRows.inverse(temp, scale);
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
                Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {
            for (int r = 0; r < rows; r++) {
                dctColumns.inverse(a[r], scale);
            }
            float[] temp = new float[rows];
            for (int c = 0; c < columns; c++) {
                for (int r = 0; r < rows; r++) {
                    temp[r] = a[r][c];
                }
                dctRows.inverse(temp, scale);
                for (int r = 0; r < rows; r++) {
                    a[r][c] = temp[r];
                }
            }
        }
    }

    private void ddxt2d_subth(final int isgn, final float[] a, final boolean scale)
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
        Future<?>[] futures = new Future[nthread];

        for (int i = 0; i < nthread; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {
                public void run()
                {
                    int idx1, idx2;
                    float[] t = new float[ntf];
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
                                dctRows.forward(t, 0, scale);
                                dctRows.forward(t, rows, scale);
                                dctRows.forward(t, 2 * rows, scale);
                                dctRows.forward(t, 3 * rows, scale);
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
                                dctRows.inverse(t, scale);
                                dctRows.inverse(t, rows, scale);
                                dctRows.inverse(t, 2 * rows, scale);
                                dctRows.inverse(t, 3 * rows, scale);
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
                            dctRows.forward(t, 0, scale);
                            dctRows.forward(t, rows, scale);
                        } else {
                            dctRows.inverse(t, 0, scale);
                            dctRows.inverse(t, rows, scale);
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
            Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void ddxt2d_subth(final int isgn, final FloatLargeArray a, final boolean scale)
    {
        int nthread = (int) min(columnsl, ConcurrencyUtils.getNumberOfThreads());
        long nt = 4 * rowsl;
        if (columnsl == 2) {
            nt >>= 1;
        } else if (columnsl < 2) {
            nt >>= 2;
        }
        final long ntf = nt;
        final long nthreads = nthread;
        Future<?>[] futures = new Future[nthread];

        for (int i = 0; i < nthread; i++) {
            final long n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {
                public void run()
                {
                    long idx1, idx2;
                    FloatLargeArray t = new FloatLargeArray(ntf);
                    if (columnsl > 2) {
                        if (isgn == -1) {
                            for (long c = 4 * n0; c < columnsl; c += 4 * nthreads) {
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = r * columnsl + c;
                                    idx2 = rowsl + r;
                                    t.setFloat(r, a.getFloat(idx1));
                                    t.setFloat(idx2, a.getFloat(idx1 + 1));
                                    t.setFloat(idx2 + rowsl, a.getFloat(idx1 + 2));
                                    t.setFloat(idx2 + 2 * rowsl, a.getFloat(idx1 + 3));
                                }
                                dctRows.forward(t, 0, scale);
                                dctRows.forward(t, rowsl, scale);
                                dctRows.forward(t, 2 * rowsl, scale);
                                dctRows.forward(t, 3 * rowsl, scale);
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = r * columnsl + c;
                                    idx2 = rowsl + r;
                                    a.setFloat(idx1, t.getFloat(r));
                                    a.setFloat(idx1 + 1, t.getFloat(idx2));
                                    a.setFloat(idx1 + 2, t.getFloat(idx2 + rowsl));
                                    a.setFloat(idx1 + 3, t.getFloat(idx2 + 2 * rowsl));
                                }
                            }
                        } else {
                            for (long c = 4 * n0; c < columnsl; c += 4 * nthreads) {
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = r * columnsl + c;
                                    idx2 = rowsl + r;
                                    t.setFloat(r, a.getFloat(idx1));
                                    t.setFloat(idx2, a.getFloat(idx1 + 1));
                                    t.setFloat(idx2 + rowsl, a.getFloat(idx1 + 2));
                                    t.setFloat(idx2 + 2 * rowsl, a.getFloat(idx1 + 3));
                                }
                                dctRows.inverse(t, scale);
                                dctRows.inverse(t, rowsl, scale);
                                dctRows.inverse(t, 2 * rowsl, scale);
                                dctRows.inverse(t, 3 * rowsl, scale);
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = r * columnsl + c;
                                    idx2 = rowsl + r;
                                    a.setFloat(idx1, t.getFloat(r));
                                    a.setFloat(idx1 + 1, t.getFloat(idx2));
                                    a.setFloat(idx1 + 2, t.getFloat(idx2 + rowsl));
                                    a.setFloat(idx1 + 3, t.getFloat(idx2 + 2 * rowsl));
                                }
                            }
                        }
                    } else if (columnsl == 2) {
                        for (long r = 0; r < rowsl; r++) {
                            idx1 = r * columnsl + 2 * n0;
                            idx2 = r;
                            t.setFloat(idx2, a.getFloat(idx1));
                            t.setFloat(idx2 + rowsl, a.getFloat(idx1 + 1));
                        }
                        if (isgn == -1) {
                            dctRows.forward(t, 0, scale);
                            dctRows.forward(t, rowsl, scale);
                        } else {
                            dctRows.inverse(t, 0, scale);
                            dctRows.inverse(t, rowsl, scale);
                        }
                        for (long r = 0; r < rowsl; r++) {
                            idx1 = r * columnsl + 2 * n0;
                            idx2 = r;
                            a.setFloat(idx1, t.getFloat(idx2));
                            a.setFloat(idx1 + 1, t.getFloat(idx2 + rowsl));
                        }
                    }
                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void ddxt2d_subth(final int isgn, final float[][] a, final boolean scale)
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
        Future<?>[] futures = new Future[nthread];

        for (int i = 0; i < nthread; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {
                public void run()
                {
                    int idx2;
                    float[] t = new float[ntf];
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
                                dctRows.forward(t, 0, scale);
                                dctRows.forward(t, rows, scale);
                                dctRows.forward(t, 2 * rows, scale);
                                dctRows.forward(t, 3 * rows, scale);
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
                                dctRows.inverse(t, 0, scale);
                                dctRows.inverse(t, rows, scale);
                                dctRows.inverse(t, 2 * rows, scale);
                                dctRows.inverse(t, 3 * rows, scale);
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
                            dctRows.forward(t, 0, scale);
                            dctRows.forward(t, rows, scale);
                        } else {
                            dctRows.inverse(t, 0, scale);
                            dctRows.inverse(t, rows, scale);
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
            Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void ddxt2d0_subth(final int isgn, final float[] a, final boolean scale)
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
                            dctColumns.forward(a, r * columns, scale);
                        }
                    } else {
                        for (int r = n0; r < rows; r += nthreads) {
                            dctColumns.inverse(a, r * columns, scale);
                        }
                    }
                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void ddxt2d0_subth(final int isgn, final FloatLargeArray a, final boolean scale)
    {
        final int nthreads = (int) (ConcurrencyUtils.getNumberOfThreads() > rowsl ? rowsl : ConcurrencyUtils.getNumberOfThreads());

        Future<?>[] futures = new Future[nthreads];

        for (int i = 0; i < nthreads; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {

                public void run()
                {
                    if (isgn == -1) {
                        for (long r = n0; r < rowsl; r += nthreads) {
                            dctColumns.forward(a, r * columnsl, scale);
                        }
                    } else {
                        for (long r = n0; r < rowsl; r += nthreads) {
                            dctColumns.inverse(a, r * columnsl, scale);
                        }
                    }
                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void ddxt2d0_subth(final int isgn, final float[][] a, final boolean scale)
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
                            dctColumns.forward(a[r], scale);
                        }
                    } else {
                        for (int r = n0; r < rows; r += nthreads) {
                            dctColumns.inverse(a[r], scale);
                        }
                    }
                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(FloatDCT_2D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void ddxt2d_sub(int isgn, float[] a, boolean scale)
    {
        int idx1, idx2;
        int nt = 4 * rows;
        if (columns == 2) {
            nt >>= 1;
        } else if (columns < 2) {
            nt >>= 2;
        }
        float[] t = new float[nt];
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
                    dctRows.forward(t, 0, scale);
                    dctRows.forward(t, rows, scale);
                    dctRows.forward(t, 2 * rows, scale);
                    dctRows.forward(t, 3 * rows, scale);
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
                    dctRows.inverse(t, 0, scale);
                    dctRows.inverse(t, rows, scale);
                    dctRows.inverse(t, 2 * rows, scale);
                    dctRows.inverse(t, 3 * rows, scale);
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
                dctRows.forward(t, 0, scale);
                dctRows.forward(t, rows, scale);
            } else {
                dctRows.inverse(t, 0, scale);
                dctRows.inverse(t, rows, scale);
            }
            for (int r = 0; r < rows; r++) {
                idx1 = r * columns;
                a[idx1] = t[r];
                a[idx1 + 1] = t[rows + r];
            }
        }
    }

    private void ddxt2d_sub(int isgn, FloatLargeArray a, boolean scale)
    {
        long idx1, idx2;
        long nt = 4 * rowsl;
        if (columnsl == 2) {
            nt >>= 1l;
        } else if (columnsl < 2) {
            nt >>= 2l;
        }
        FloatLargeArray t = new FloatLargeArray(nt);
        if (columnsl > 2) {
            if (isgn == -1) {
                for (long c = 0; c < columnsl; c += 4) {
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = r * columnsl + c;
                        idx2 = rowsl + r;
                        t.setFloat(r, a.getFloat(idx1));
                        t.setFloat(idx2, a.getFloat(idx1 + 1));
                        t.setFloat(idx2 + rowsl, a.getFloat(idx1 + 2));
                        t.setFloat(idx2 + 2 * rowsl, a.getFloat(idx1 + 3));
                    }
                    dctRows.forward(t, 0, scale);
                    dctRows.forward(t, rowsl, scale);
                    dctRows.forward(t, 2 * rowsl, scale);
                    dctRows.forward(t, 3 * rowsl, scale);
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = r * columnsl + c;
                        idx2 = rowsl + r;
                        a.setFloat(idx1, t.getFloat(r));
                        a.setFloat(idx1 + 1, t.getFloat(idx2));
                        a.setFloat(idx1 + 2, t.getFloat(idx2 + rowsl));
                        a.setFloat(idx1 + 3, t.getFloat(idx2 + 2 * rowsl));
                    }
                }
            } else {
                for (long c = 0; c < columnsl; c += 4) {
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = r * columnsl + c;
                        idx2 = rowsl + r;
                        t.setFloat(r, a.getFloat(idx1));
                        t.setFloat(idx2, a.getFloat(idx1 + 1));
                        t.setFloat(idx2 + rowsl, a.getFloat(idx1 + 2));
                        t.setFloat(idx2 + 2 * rowsl, a.getFloat(idx1 + 3));
                    }
                    dctRows.inverse(t, 0, scale);
                    dctRows.inverse(t, rowsl, scale);
                    dctRows.inverse(t, 2 * rowsl, scale);
                    dctRows.inverse(t, 3 * rowsl, scale);
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = r * columnsl + c;
                        idx2 = rowsl + r;
                        a.setFloat(idx1, t.getFloat(r));
                        a.setFloat(idx1 + 1, t.getFloat(idx2));
                        a.setFloat(idx1 + 2, t.getFloat(idx2 + rowsl));
                        a.setFloat(idx1 + 3, t.getFloat(idx2 + 2 * rowsl));
                    }
                }
            }
        } else if (columnsl == 2) {
            for (long r = 0; r < rowsl; r++) {
                idx1 = r * columnsl;
                t.setFloat(r, a.getFloat(idx1));
                t.setFloat(rowsl + r, a.getFloat(idx1 + 1));
            }
            if (isgn == -1) {
                dctRows.forward(t, 0, scale);
                dctRows.forward(t, rowsl, scale);
            } else {
                dctRows.inverse(t, 0, scale);
                dctRows.inverse(t, rowsl, scale);
            }
            for (long r = 0; r < rowsl; r++) {
                idx1 = r * columnsl;
                a.setFloat(idx1, t.getFloat(r));
                a.setFloat(idx1 + 1, t.getFloat(rowsl + r));
            }
        }
    }

    private void ddxt2d_sub(int isgn, float[][] a, boolean scale)
    {
        int idx2;
        int nt = 4 * rows;
        if (columns == 2) {
            nt >>= 1;
        } else if (columns < 2) {
            nt >>= 2;
        }
        float[] t = new float[nt];
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
                    dctRows.forward(t, 0, scale);
                    dctRows.forward(t, rows, scale);
                    dctRows.forward(t, 2 * rows, scale);
                    dctRows.forward(t, 3 * rows, scale);
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
                    dctRows.inverse(t, 0, scale);
                    dctRows.inverse(t, rows, scale);
                    dctRows.inverse(t, 2 * rows, scale);
                    dctRows.inverse(t, 3 * rows, scale);
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
                dctRows.forward(t, 0, scale);
                dctRows.forward(t, rows, scale);
            } else {
                dctRows.inverse(t, 0, scale);
                dctRows.inverse(t, rows, scale);
            }
            for (int r = 0; r < rows; r++) {
                a[r][0] = t[r];
                a[r][1] = t[rows + r];
            }
        }
    }
}
