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
import java.util.concurrent.ExecutionException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.jtransforms.utils.CommonUtils;
import pl.edu.icm.jlargearrays.ConcurrencyUtils;
import pl.edu.icm.jlargearrays.DoubleLargeArray;
import pl.edu.icm.jlargearrays.LargeArray;

/**
 * Computes 3D Discrete Sine Transform (DST) of double precision data. The sizes
 * of all three dimensions can be arbitrary numbers. This is a parallel
 * implementation optimized for SMP systems.<br>
 * <br>
 * Part of code is derived from General Purpose FFT Package written by Takuya
 * Ooura (http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html)
 *  
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class DoubleDST_3D
{

    private int slices;

    private long slicesl;

    private int rows;

    private long rowsl;

    private int columns;

    private long columnsl;

    private int sliceStride;

    private long sliceStridel;

    private int rowStride;

    private long rowStridel;

    private DoubleDST_1D dstSlices, dstRows, dstColumns;

    private boolean isPowerOfTwo = false;

    private boolean useThreads = false;

    /**
     * Creates new instance of DoubleDST_3D.
     *  
     * @param slices  number of slices
     * @param rows    number of rows
     * @param columns number of columns
     */
    public DoubleDST_3D(long slices, long rows, long columns)
    {
        if (slices <= 1 || rows <= 1 || columns <= 1) {
            throw new IllegalArgumentException("slices, rows and columns must be greater than 1");
        }
        this.slices = (int) slices;
        this.rows = (int) rows;
        this.columns = (int) columns;
        this.slicesl = slices;
        this.rowsl = rows;
        this.columnsl = columns;
        this.sliceStride = (int) (rows * columns);
        this.rowStride = (int) columns;
        this.sliceStridel = rows * columns;
        this.rowStridel = columns;
        if (slices * rows * columns >= CommonUtils.getThreadsBeginN_3D()) {
            this.useThreads = true;
        }
        if (CommonUtils.isPowerOf2(slices) && CommonUtils.isPowerOf2(rows) && CommonUtils.isPowerOf2(columns)) {
            isPowerOfTwo = true;
        }
        CommonUtils.setUseLargeArrays(slices * rows * columns > LargeArray.getMaxSizeOf32bitArray());
        dstSlices = new DoubleDST_1D(slices);
        if (slices == rows) {
            dstRows = dstSlices;
        } else {
            dstRows = new DoubleDST_1D(rows);
        }
        if (slices == columns) {
            dstColumns = dstSlices;
        } else if (rows == columns) {
            dstColumns = dstRows;
        } else {
            dstColumns = new DoubleDST_1D(columns);
        }
    }

    /**
     * Computes the 3D forward DST (DST-II) leaving the result in <code>a</code>
     * . The data is stored in 1D array addressed in slice-major, then
     * row-major, then column-major, in order of significance, i.e. the element
     * (i,j,k) of 3D array x[slices][rows][columns] is stored in a[i*sliceStride
     * + j*rowStride + k], where sliceStride = rows * columns and rowStride =
     * columns.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void forward(final double[] a, final boolean scale)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            if ((nthreads > 1) && useThreads) {
                ddxt3da_subth(-1, a, scale);
                ddxt3db_subth(-1, a, scale);
            } else {
                ddxt3da_sub(-1, a, scale);
                ddxt3db_sub(-1, a, scale);
            }
        } else if ((nthreads > 1) && useThreads && (slices >= nthreads) && (rows >= nthreads) && (columns >= nthreads)) {
            Future<?>[] futures = new Future[nthreads];
            int p = slices / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (int s = firstSlice; s < lastSlice; s++) {
                            int idx1 = s * sliceStride;
                            for (int r = 0; r < rows; r++) {
                                dstColumns.forward(a, idx1 + r * rowStride, scale);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        double[] temp = new double[rows];
                        for (int s = firstSlice; s < lastSlice; s++) {
                            int idx1 = s * sliceStride;
                            for (int c = 0; c < columns; c++) {
                                for (int r = 0; r < rows; r++) {
                                    int idx3 = idx1 + r * rowStride + c;
                                    temp[r] = a[idx3];
                                }
                                dstRows.forward(temp, scale);
                                for (int r = 0; r < rows; r++) {
                                    int idx3 = idx1 + r * rowStride + c;
                                    a[idx3] = temp[r];
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            p = rows / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstRow = l * p;
                final int lastRow = (l == (nthreads - 1)) ? rows : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        double[] temp = new double[slices];
                        for (int r = firstRow; r < lastRow; r++) {
                            int idx1 = r * rowStride;
                            for (int c = 0; c < columns; c++) {
                                for (int s = 0; s < slices; s++) {
                                    int idx3 = s * sliceStride + idx1 + c;
                                    temp[s] = a[idx3];
                                }
                                dstSlices.forward(temp, scale);
                                for (int s = 0; s < slices; s++) {
                                    int idx3 = s * sliceStride + idx1 + c;
                                    a[idx3] = temp[s];
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

        } else {
            for (int s = 0; s < slices; s++) {
                int idx1 = s * sliceStride;
                for (int r = 0; r < rows; r++) {
                    dstColumns.forward(a, idx1 + r * rowStride, scale);
                }
            }
            double[] temp = new double[rows];
            for (int s = 0; s < slices; s++) {
                int idx1 = s * sliceStride;
                for (int c = 0; c < columns; c++) {
                    for (int r = 0; r < rows; r++) {
                        int idx3 = idx1 + r * rowStride + c;
                        temp[r] = a[idx3];
                    }
                    dstRows.forward(temp, scale);
                    for (int r = 0; r < rows; r++) {
                        int idx3 = idx1 + r * rowStride + c;
                        a[idx3] = temp[r];
                    }
                }
            }
            temp = new double[slices];
            for (int r = 0; r < rows; r++) {
                int idx1 = r * rowStride;
                for (int c = 0; c < columns; c++) {
                    for (int s = 0; s < slices; s++) {
                        int idx3 = s * sliceStride + idx1 + c;
                        temp[s] = a[idx3];
                    }
                    dstSlices.forward(temp, scale);
                    for (int s = 0; s < slices; s++) {
                        int idx3 = s * sliceStride + idx1 + c;
                        a[idx3] = temp[s];
                    }
                }
            }
        }
    }

    /**
     * Computes the 3D forward DST (DST-II) leaving the result in <code>a</code>
     * . The data is stored in 1D array addressed in slice-major, then
     * row-major, then column-major, in order of significance, i.e. the element
     * (i,j,k) of 3D array x[slices][rows][columns] is stored in a[i*sliceStride
     * + j*rowStride + k], where sliceStride = rows * columns and rowStride =
     * columns.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void forward(final DoubleLargeArray a, final boolean scale)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            if ((nthreads > 1) && useThreads) {
                ddxt3da_subth(-1, a, scale);
                ddxt3db_subth(-1, a, scale);
            } else {
                ddxt3da_sub(-1, a, scale);
                ddxt3db_sub(-1, a, scale);
            }
        } else if ((nthreads > 1) && useThreads && (slicesl >= nthreads) && (rowsl >= nthreads) && (columnsl >= nthreads)) {
            Future<?>[] futures = new Future[nthreads];
            long p = slicesl / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final long firstSlice = l * p;
                final long lastSlice = (l == (nthreads - 1)) ? slicesl : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (long s = firstSlice; s < lastSlice; s++) {
                            long idx1 = s * sliceStridel;
                            for (long r = 0; r < rowsl; r++) {
                                dstColumns.forward(a, idx1 + r * rowStridel, scale);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int l = 0; l < nthreads; l++) {
                final long firstSlice = l * p;
                final long lastSlice = (l == (nthreads - 1)) ? slicesl : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        DoubleLargeArray temp = new DoubleLargeArray(rowsl, false);
                        for (long s = firstSlice; s < lastSlice; s++) {
                            long idx1 = s * sliceStridel;
                            for (long c = 0; c < columnsl; c++) {
                                for (long r = 0; r < rowsl; r++) {
                                    long idx3 = idx1 + r * rowStridel + c;
                                    temp.setDouble(r, a.getDouble(idx3));
                                }
                                dstRows.forward(temp, scale);
                                for (long r = 0; r < rowsl; r++) {
                                    long idx3 = idx1 + r * rowStridel + c;
                                    a.setDouble(idx3, temp.getDouble(r));
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            p = rowsl / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final long firstRow = l * p;
                final long lastRow = (l == (nthreads - 1)) ? rowsl : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        DoubleLargeArray temp = new DoubleLargeArray(slicesl, false);
                        for (long r = firstRow; r < lastRow; r++) {
                            long idx1 = r * rowStridel;
                            for (long c = 0; c < columnsl; c++) {
                                for (long s = 0; s < slicesl; s++) {
                                    long idx3 = s * sliceStridel + idx1 + c;
                                    temp.setDouble(s, a.getDouble(idx3));
                                }
                                dstSlices.forward(temp, scale);
                                for (long s = 0; s < slicesl; s++) {
                                    long idx3 = s * sliceStridel + idx1 + c;
                                    a.setDouble(idx3, temp.getDouble(s));
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

        } else {
            for (long s = 0; s < slicesl; s++) {
                long idx1 = s * sliceStridel;
                for (long r = 0; r < rowsl; r++) {
                    dstColumns.forward(a, idx1 + r * rowStridel, scale);
                }
            }
            DoubleLargeArray temp = new DoubleLargeArray(rowsl, false);
            for (long s = 0; s < slicesl; s++) {
                long idx1 = s * sliceStridel;
                for (long c = 0; c < columnsl; c++) {
                    for (long r = 0; r < rowsl; r++) {
                        long idx3 = idx1 + r * rowStridel + c;
                        temp.setDouble(r, a.getDouble(idx3));
                    }
                    dstRows.forward(temp, scale);
                    for (long r = 0; r < rowsl; r++) {
                        long idx3 = idx1 + r * rowStridel + c;
                        a.setDouble(idx3, temp.getDouble(r));
                    }
                }
            }
            temp = new DoubleLargeArray(slicesl, false);
            for (long r = 0; r < rowsl; r++) {
                long idx1 = r * rowStridel;
                for (long c = 0; c < columnsl; c++) {
                    for (long s = 0; s < slicesl; s++) {
                        long idx3 = s * sliceStridel + idx1 + c;
                        temp.setDouble(s, a.getDouble(idx3));
                    }
                    dstSlices.forward(temp, scale);
                    for (long s = 0; s < slicesl; s++) {
                        long idx3 = s * sliceStridel + idx1 + c;
                        a.setDouble(idx3, temp.getDouble(s));
                    }
                }
            }
        }
    }

    /**
     * Computes the 3D forward DST (DST-II) leaving the result in <code>a</code>
     * . The data is stored in 3D array.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void forward(final double[][][] a, final boolean scale)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            if ((nthreads > 1) && useThreads) {
                ddxt3da_subth(-1, a, scale);
                ddxt3db_subth(-1, a, scale);
            } else {
                ddxt3da_sub(-1, a, scale);
                ddxt3db_sub(-1, a, scale);
            }
        } else if ((nthreads > 1) && useThreads && (slices >= nthreads) && (rows >= nthreads) && (columns >= nthreads)) {
            Future<?>[] futures = new Future[nthreads];
            int p = slices / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                dstColumns.forward(a[s][r], scale);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        double[] temp = new double[rows];
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int c = 0; c < columns; c++) {
                                for (int r = 0; r < rows; r++) {
                                    temp[r] = a[s][r][c];
                                }
                                dstRows.forward(temp, scale);
                                for (int r = 0; r < rows; r++) {
                                    a[s][r][c] = temp[r];
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            p = rows / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstRow = l * p;
                final int lastRow = (l == (nthreads - 1)) ? rows : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        double[] temp = new double[slices];
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                for (int s = 0; s < slices; s++) {
                                    temp[s] = a[s][r][c];
                                }
                                dstSlices.forward(temp, scale);
                                for (int s = 0; s < slices; s++) {
                                    a[s][r][c] = temp[s];
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

        } else {
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    dstColumns.forward(a[s][r], scale);
                }
            }
            double[] temp = new double[rows];
            for (int s = 0; s < slices; s++) {
                for (int c = 0; c < columns; c++) {
                    for (int r = 0; r < rows; r++) {
                        temp[r] = a[s][r][c];
                    }
                    dstRows.forward(temp, scale);
                    for (int r = 0; r < rows; r++) {
                        a[s][r][c] = temp[r];
                    }
                }
            }
            temp = new double[slices];
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    for (int s = 0; s < slices; s++) {
                        temp[s] = a[s][r][c];
                    }
                    dstSlices.forward(temp, scale);
                    for (int s = 0; s < slices; s++) {
                        a[s][r][c] = temp[s];
                    }
                }
            }
        }
    }

    /**
     * Computes the 3D inverse DST (DST-III) leaving the result in
     * <code>a</code>. The data is stored in 1D array addressed in slice-major,
     * then row-major, then column-major, in order of significance, i.e. the
     * element (i,j,k) of 3D array x[slices][rows][columns] is stored in
     * a[i*sliceStride + j*rowStride + k], where sliceStride = rows * columns
     * and rowStride = columns.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void inverse(final double[] a, final boolean scale)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            if ((nthreads > 1) && useThreads) {
                ddxt3da_subth(1, a, scale);
                ddxt3db_subth(1, a, scale);
            } else {
                ddxt3da_sub(1, a, scale);
                ddxt3db_sub(1, a, scale);
            }
        } else if ((nthreads > 1) && useThreads && (slices >= nthreads) && (rows >= nthreads) && (columns >= nthreads)) {
            Future<?>[] futures = new Future[nthreads];
            int p = slices / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (int s = firstSlice; s < lastSlice; s++) {
                            int idx1 = s * sliceStride;
                            for (int r = 0; r < rows; r++) {
                                dstColumns.inverse(a, idx1 + r * rowStride, scale);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        double[] temp = new double[rows];
                        for (int s = firstSlice; s < lastSlice; s++) {
                            int idx1 = s * sliceStride;
                            for (int c = 0; c < columns; c++) {
                                for (int r = 0; r < rows; r++) {
                                    int idx3 = idx1 + r * rowStride + c;
                                    temp[r] = a[idx3];
                                }
                                dstRows.inverse(temp, scale);
                                for (int r = 0; r < rows; r++) {
                                    int idx3 = idx1 + r * rowStride + c;
                                    a[idx3] = temp[r];
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            p = rows / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstRow = l * p;
                final int lastRow = (l == (nthreads - 1)) ? rows : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        double[] temp = new double[slices];
                        for (int r = firstRow; r < lastRow; r++) {
                            int idx1 = r * rowStride;
                            for (int c = 0; c < columns; c++) {
                                for (int s = 0; s < slices; s++) {
                                    int idx3 = s * sliceStride + idx1 + c;
                                    temp[s] = a[idx3];
                                }
                                dstSlices.inverse(temp, scale);
                                for (int s = 0; s < slices; s++) {
                                    int idx3 = s * sliceStride + idx1 + c;
                                    a[idx3] = temp[s];
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

        } else {
            for (int s = 0; s < slices; s++) {
                int idx1 = s * sliceStride;
                for (int r = 0; r < rows; r++) {
                    dstColumns.inverse(a, idx1 + r * rowStride, scale);
                }
            }
            double[] temp = new double[rows];
            for (int s = 0; s < slices; s++) {
                int idx1 = s * sliceStride;
                for (int c = 0; c < columns; c++) {
                    for (int r = 0; r < rows; r++) {
                        int idx3 = idx1 + r * rowStride + c;
                        temp[r] = a[idx3];
                    }
                    dstRows.inverse(temp, scale);
                    for (int r = 0; r < rows; r++) {
                        int idx3 = idx1 + r * rowStride + c;
                        a[idx3] = temp[r];
                    }
                }
            }
            temp = new double[slices];
            for (int r = 0; r < rows; r++) {
                int idx1 = r * rowStride;
                for (int c = 0; c < columns; c++) {
                    for (int s = 0; s < slices; s++) {
                        int idx3 = s * sliceStride + idx1 + c;
                        temp[s] = a[idx3];
                    }
                    dstSlices.inverse(temp, scale);
                    for (int s = 0; s < slices; s++) {
                        int idx3 = s * sliceStride + idx1 + c;
                        a[idx3] = temp[s];
                    }
                }
            }
        }

    }

    /**
     * Computes the 3D inverse DST (DST-III) leaving the result in
     * <code>a</code>. The data is stored in 1D array addressed in slice-major,
     * then row-major, then column-major, in order of significance, i.e. the
     * element (i,j,k) of 3D array x[slices][rows][columns] is stored in
     * a[i*sliceStride + j*rowStride + k], where sliceStride = rows * columns
     * and rowStride = columns.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void inverse(final DoubleLargeArray a, final boolean scale)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            if ((nthreads > 1) && useThreads) {
                ddxt3da_subth(1, a, scale);
                ddxt3db_subth(1, a, scale);
            } else {
                ddxt3da_sub(1, a, scale);
                ddxt3db_sub(1, a, scale);
            }
        } else if ((nthreads > 1) && useThreads && (slicesl >= nthreads) && (rowsl >= nthreads) && (columnsl >= nthreads)) {
            Future<?>[] futures = new Future[nthreads];
            long p = slicesl / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final long firstSlice = l * p;
                final long lastSlice = (l == (nthreads - 1)) ? slicesl : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (long s = firstSlice; s < lastSlice; s++) {
                            long idx1 = s * sliceStride;
                            for (long r = 0; r < rowsl; r++) {
                                dstColumns.inverse(a, idx1 + r * rowStride, scale);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int l = 0; l < nthreads; l++) {
                final long firstSlice = l * p;
                final long lastSlice = (l == (nthreads - 1)) ? slicesl : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        DoubleLargeArray temp = new DoubleLargeArray(rowsl, false);
                        for (long s = firstSlice; s < lastSlice; s++) {
                            long idx1 = s * sliceStride;
                            for (long c = 0; c < columnsl; c++) {
                                for (long r = 0; r < rowsl; r++) {
                                    long idx3 = idx1 + r * rowStride + c;
                                    temp.setDouble(r, a.getDouble(idx3));
                                }
                                dstRows.inverse(temp, scale);
                                for (long r = 0; r < rowsl; r++) {
                                    long idx3 = idx1 + r * rowStride + c;
                                    a.setDouble(idx3, temp.getDouble(r));
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            p = rowsl / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final long firstRow = l * p;
                final long lastRow = (l == (nthreads - 1)) ? rowsl : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        DoubleLargeArray temp = new DoubleLargeArray(slicesl, false);
                        for (long r = firstRow; r < lastRow; r++) {
                            long idx1 = r * rowStride;
                            for (long c = 0; c < columnsl; c++) {
                                for (long s = 0; s < slicesl; s++) {
                                    long idx3 = s * sliceStride + idx1 + c;
                                    temp.setDouble(s, a.getDouble(idx3));
                                }
                                dstSlices.inverse(temp, scale);
                                for (long s = 0; s < slicesl; s++) {
                                    long idx3 = s * sliceStride + idx1 + c;
                                    a.setDouble(idx3, temp.getDouble(s));
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

        } else {
            for (long s = 0; s < slicesl; s++) {
                long idx1 = s * sliceStride;
                for (long r = 0; r < rowsl; r++) {
                    dstColumns.inverse(a, idx1 + r * rowStride, scale);
                }
            }
            DoubleLargeArray temp = new DoubleLargeArray(rowsl, false);
            for (long s = 0; s < slicesl; s++) {
                long idx1 = s * sliceStride;
                for (long c = 0; c < columnsl; c++) {
                    for (long r = 0; r < rowsl; r++) {
                        long idx3 = idx1 + r * rowStride + c;
                        temp.setDouble(r, a.getDouble(idx3));
                    }
                    dstRows.inverse(temp, scale);
                    for (long r = 0; r < rowsl; r++) {
                        long idx3 = idx1 + r * rowStride + c;
                        a.setDouble(idx3, temp.getDouble(r));
                    }
                }
            }
            temp = new DoubleLargeArray(slicesl, false);
            for (long r = 0; r < rowsl; r++) {
                long idx1 = r * rowStride;
                for (long c = 0; c < columnsl; c++) {
                    for (long s = 0; s < slicesl; s++) {
                        long idx3 = s * sliceStride + idx1 + c;
                        temp.setDouble(s, a.getDouble(idx3));
                    }
                    dstSlices.inverse(temp, scale);
                    for (long s = 0; s < slicesl; s++) {
                        long idx3 = s * sliceStride + idx1 + c;
                        a.setDouble(idx3, temp.getDouble(s));
                    }
                }
            }
        }

    }

    /**
     * Computes the 3D inverse DST (DST-III) leaving the result in
     * <code>a</code>. The data is stored in 3D array.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void inverse(final double[][][] a, final boolean scale)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            if ((nthreads > 1) && useThreads) {
                ddxt3da_subth(1, a, scale);
                ddxt3db_subth(1, a, scale);
            } else {
                ddxt3da_sub(1, a, scale);
                ddxt3db_sub(1, a, scale);
            }
        } else if ((nthreads > 1) && useThreads && (slices >= nthreads) && (rows >= nthreads) && (columns >= nthreads)) {
            Future<?>[] futures = new Future[nthreads];
            int p = slices / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int r = 0; r < rows; r++) {
                                dstColumns.inverse(a[s][r], scale);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        double[] temp = new double[rows];
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int c = 0; c < columns; c++) {
                                for (int r = 0; r < rows; r++) {
                                    temp[r] = a[s][r][c];
                                }
                                dstRows.inverse(temp, scale);
                                for (int r = 0; r < rows; r++) {
                                    a[s][r][c] = temp[r];
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            p = rows / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstRow = l * p;
                final int lastRow = (l == (nthreads - 1)) ? rows : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        double[] temp = new double[slices];
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                for (int s = 0; s < slices; s++) {
                                    temp[s] = a[s][r][c];
                                }
                                dstSlices.inverse(temp, scale);
                                for (int s = 0; s < slices; s++) {
                                    a[s][r][c] = temp[s];
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

        } else {
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    dstColumns.inverse(a[s][r], scale);
                }
            }
            double[] temp = new double[rows];
            for (int s = 0; s < slices; s++) {
                for (int c = 0; c < columns; c++) {
                    for (int r = 0; r < rows; r++) {
                        temp[r] = a[s][r][c];
                    }
                    dstRows.inverse(temp, scale);
                    for (int r = 0; r < rows; r++) {
                        a[s][r][c] = temp[r];
                    }
                }
            }
            temp = new double[slices];
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    for (int s = 0; s < slices; s++) {
                        temp[s] = a[s][r][c];
                    }
                    dstSlices.inverse(temp, scale);
                    for (int s = 0; s < slices; s++) {
                        a[s][r][c] = temp[s];
                    }
                }
            }
        }
    }

    private void ddxt3da_sub(int isgn, double[] a, boolean scale)
    {
        int idx0, idx1, idx2;
        int nt = 4 * rows;
        if (columns == 2) {
            nt >>= 1;
        }
        double[] t = new double[nt];
        if (isgn == -1) {
            for (int s = 0; s < slices; s++) {
                idx0 = s * sliceStride;
                for (int r = 0; r < rows; r++) {
                    dstColumns.forward(a, idx0 + r * rowStride, scale);
                }
                if (columns > 2) {
                    for (int c = 0; c < columns; c += 4) {
                        for (int r = 0; r < rows; r++) {
                            idx1 = idx0 + r * rowStride + c;
                            idx2 = rows + r;
                            t[r] = a[idx1];
                            t[idx2] = a[idx1 + 1];
                            t[idx2 + rows] = a[idx1 + 2];
                            t[idx2 + 2 * rows] = a[idx1 + 3];
                        }
                        dstRows.forward(t, 0, scale);
                        dstRows.forward(t, rows, scale);
                        dstRows.forward(t, 2 * rows, scale);
                        dstRows.forward(t, 3 * rows, scale);
                        for (int r = 0; r < rows; r++) {
                            idx1 = idx0 + r * rowStride + c;
                            idx2 = rows + r;
                            a[idx1] = t[r];
                            a[idx1 + 1] = t[idx2];
                            a[idx1 + 2] = t[idx2 + rows];
                            a[idx1 + 3] = t[idx2 + 2 * rows];
                        }
                    }
                } else if (columns == 2) {
                    for (int r = 0; r < rows; r++) {
                        idx1 = idx0 + r * rowStride;
                        t[r] = a[idx1];
                        t[rows + r] = a[idx1 + 1];
                    }
                    dstRows.forward(t, 0, scale);
                    dstRows.forward(t, rows, scale);
                    for (int r = 0; r < rows; r++) {
                        idx1 = idx0 + r * rowStride;
                        a[idx1] = t[r];
                        a[idx1 + 1] = t[rows + r];
                    }
                }
            }
        } else {
            for (int s = 0; s < slices; s++) {
                idx0 = s * sliceStride;
                for (int r = 0; r < rows; r++) {
                    dstColumns.inverse(a, idx0 + r * rowStride, scale);
                }
                if (columns > 2) {
                    for (int c = 0; c < columns; c += 4) {
                        for (int r = 0; r < rows; r++) {
                            idx1 = idx0 + r * rowStride + c;
                            idx2 = rows + r;
                            t[r] = a[idx1];
                            t[idx2] = a[idx1 + 1];
                            t[idx2 + rows] = a[idx1 + 2];
                            t[idx2 + 2 * rows] = a[idx1 + 3];
                        }
                        dstRows.inverse(t, 0, scale);
                        dstRows.inverse(t, rows, scale);
                        dstRows.inverse(t, 2 * rows, scale);
                        dstRows.inverse(t, 3 * rows, scale);
                        for (int r = 0; r < rows; r++) {
                            idx1 = idx0 + r * rowStride + c;
                            idx2 = rows + r;
                            a[idx1] = t[r];
                            a[idx1 + 1] = t[idx2];
                            a[idx1 + 2] = t[idx2 + rows];
                            a[idx1 + 3] = t[idx2 + 2 * rows];
                        }
                    }
                } else if (columns == 2) {
                    for (int r = 0; r < rows; r++) {
                        idx1 = idx0 + r * rowStride;
                        t[r] = a[idx1];
                        t[rows + r] = a[idx1 + 1];
                    }
                    dstRows.inverse(t, 0, scale);
                    dstRows.inverse(t, rows, scale);
                    for (int r = 0; r < rows; r++) {
                        idx1 = idx0 + r * rowStride;
                        a[idx1] = t[r];
                        a[idx1 + 1] = t[rows + r];
                    }
                }
            }
        }
    }

    private void ddxt3da_sub(int isgn, DoubleLargeArray a, boolean scale)
    {
        long idx0, idx1, idx2;
        long nt = 4 * rowsl;
        if (columnsl == 2) {
            nt >>= 1;
        }
        DoubleLargeArray t = new DoubleLargeArray(nt);
        if (isgn == -1) {
            for (long s = 0; s < slicesl; s++) {
                idx0 = s * sliceStridel;
                for (long r = 0; r < rowsl; r++) {
                    dstColumns.forward(a, idx0 + r * rowStride, scale);
                }
                if (columnsl > 2) {
                    for (long c = 0; c < columnsl; c += 4) {
                        for (long r = 0; r < rowsl; r++) {
                            idx1 = idx0 + r * rowStridel + c;
                            idx2 = rowsl + r;
                            t.setDouble(r, a.getDouble(idx1));
                            t.setDouble(idx2, a.getDouble(idx1 + 1));
                            t.setDouble(idx2 + rowsl, a.getDouble(idx1 + 2));
                            t.setDouble(idx2 + 2 * rowsl, a.getDouble(idx1 + 3));
                        }
                        dstRows.forward(t, 0, scale);
                        dstRows.forward(t, rowsl, scale);
                        dstRows.forward(t, 2 * rowsl, scale);
                        dstRows.forward(t, 3 * rowsl, scale);
                        for (long r = 0; r < rowsl; r++) {
                            idx1 = idx0 + r * rowStridel + c;
                            idx2 = rowsl + r;
                            a.setDouble(idx1, t.getDouble(r));
                            a.setDouble(idx1 + 1, t.getDouble(idx2));
                            a.setDouble(idx1 + 2, t.getDouble(idx2 + rowsl));
                            a.setDouble(idx1 + 3, t.getDouble(idx2 + 2 * rowsl));
                        }
                    }
                } else if (columnsl == 2) {
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = idx0 + r * rowStridel;
                        t.setDouble(r, a.getDouble(idx1));
                        t.setDouble(rowsl + r, a.getDouble(idx1 + 1));
                    }
                    dstRows.forward(t, 0, scale);
                    dstRows.forward(t, rowsl, scale);
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = idx0 + r * rowStridel;
                        a.setDouble(idx1, t.getDouble(r));
                        a.setDouble(idx1 + 1, t.getDouble(rowsl + r));
                    }
                }
            }
        } else {
            for (long s = 0; s < slicesl; s++) {
                idx0 = s * sliceStridel;
                for (long r = 0; r < rowsl; r++) {
                    dstColumns.inverse(a, idx0 + r * rowStridel, scale);
                }
                if (columnsl > 2) {
                    for (long c = 0; c < columnsl; c += 4) {
                        for (long r = 0; r < rowsl; r++) {
                            idx1 = idx0 + r * rowStridel + c;
                            idx2 = rowsl + r;
                            t.setDouble(r, a.getDouble(idx1));
                            t.setDouble(idx2, a.getDouble(idx1 + 1));
                            t.setDouble(idx2 + rowsl, a.getDouble(idx1 + 2));
                            t.setDouble(idx2 + 2 * rowsl, a.getDouble(idx1 + 3));
                        }
                        dstRows.inverse(t, 0, scale);
                        dstRows.inverse(t, rowsl, scale);
                        dstRows.inverse(t, 2 * rowsl, scale);
                        dstRows.inverse(t, 3 * rowsl, scale);
                        for (long r = 0; r < rowsl; r++) {
                            idx1 = idx0 + r * rowStridel + c;
                            idx2 = rowsl + r;
                            a.setDouble(idx1, t.getDouble(r));
                            a.setDouble(idx1 + 1, t.getDouble(idx2));
                            a.setDouble(idx1 + 2, t.getDouble(idx2 + rowsl));
                            a.setDouble(idx1 + 3, t.getDouble(idx2 + 2 * rowsl));
                        }
                    }
                } else if (columnsl == 2) {
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = idx0 + r * rowStridel;
                        t.setDouble(r, a.getDouble(idx1));
                        t.setDouble(rowsl + r, a.getDouble(idx1 + 1));
                    }
                    dstRows.inverse(t, 0, scale);
                    dstRows.inverse(t, rowsl, scale);
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = idx0 + r * rowStridel;
                        a.setDouble(idx1, t.getDouble(r));
                        a.setDouble(idx1 + 1, t.getDouble(rowsl + r));
                    }
                }
            }
        }
    }

    private void ddxt3da_sub(int isgn, double[][][] a, boolean scale)
    {
        int idx2;
        int nt = 4 * rows;
        if (columns == 2) {
            nt >>= 1;
        }
        double[] t = new double[nt];

        if (isgn == -1) {
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    dstColumns.forward(a[s][r], scale);
                }
                if (columns > 2) {
                    for (int c = 0; c < columns; c += 4) {
                        for (int r = 0; r < rows; r++) {
                            idx2 = rows + r;
                            t[r] = a[s][r][c];
                            t[idx2] = a[s][r][c + 1];
                            t[idx2 + rows] = a[s][r][c + 2];
                            t[idx2 + 2 * rows] = a[s][r][c + 3];
                        }
                        dstRows.forward(t, 0, scale);
                        dstRows.forward(t, rows, scale);
                        dstRows.forward(t, 2 * rows, scale);
                        dstRows.forward(t, 3 * rows, scale);
                        for (int r = 0; r < rows; r++) {
                            idx2 = rows + r;
                            a[s][r][c] = t[r];
                            a[s][r][c + 1] = t[idx2];
                            a[s][r][c + 2] = t[idx2 + rows];
                            a[s][r][c + 3] = t[idx2 + 2 * rows];
                        }
                    }
                } else if (columns == 2) {
                    for (int r = 0; r < rows; r++) {
                        t[r] = a[s][r][0];
                        t[rows + r] = a[s][r][1];
                    }
                    dstRows.forward(t, 0, scale);
                    dstRows.forward(t, rows, scale);
                    for (int r = 0; r < rows; r++) {
                        a[s][r][0] = t[r];
                        a[s][r][1] = t[rows + r];
                    }
                }
            }
        } else {
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    dstColumns.inverse(a[s][r], scale);
                }
                if (columns > 2) {
                    for (int c = 0; c < columns; c += 4) {
                        for (int r = 0; r < rows; r++) {
                            idx2 = rows + r;
                            t[r] = a[s][r][c];
                            t[idx2] = a[s][r][c + 1];
                            t[idx2 + rows] = a[s][r][c + 2];
                            t[idx2 + 2 * rows] = a[s][r][c + 3];
                        }
                        dstRows.inverse(t, 0, scale);
                        dstRows.inverse(t, rows, scale);
                        dstRows.inverse(t, 2 * rows, scale);
                        dstRows.inverse(t, 3 * rows, scale);
                        for (int r = 0; r < rows; r++) {
                            idx2 = rows + r;
                            a[s][r][c] = t[r];
                            a[s][r][c + 1] = t[idx2];
                            a[s][r][c + 2] = t[idx2 + rows];
                            a[s][r][c + 3] = t[idx2 + 2 * rows];
                        }
                    }
                } else if (columns == 2) {
                    for (int r = 0; r < rows; r++) {
                        t[r] = a[s][r][0];
                        t[rows + r] = a[s][r][1];
                    }
                    dstRows.inverse(t, 0, scale);
                    dstRows.inverse(t, rows, scale);
                    for (int r = 0; r < rows; r++) {
                        a[s][r][0] = t[r];
                        a[s][r][1] = t[rows + r];
                    }
                }
            }
        }
    }

    private void ddxt3db_sub(int isgn, double[] a, boolean scale)
    {
        int idx0, idx1, idx2;
        int nt = 4 * slices;
        if (columns == 2) {
            nt >>= 1;
        }
        double[] t = new double[nt];
        if (isgn == -1) {
            if (columns > 2) {
                for (int r = 0; r < rows; r++) {
                    idx0 = r * rowStride;
                    for (int c = 0; c < columns; c += 4) {
                        for (int s = 0; s < slices; s++) {
                            idx1 = s * sliceStride + idx0 + c;
                            idx2 = slices + s;
                            t[s] = a[idx1];
                            t[idx2] = a[idx1 + 1];
                            t[idx2 + slices] = a[idx1 + 2];
                            t[idx2 + 2 * slices] = a[idx1 + 3];
                        }
                        dstSlices.forward(t, 0, scale);
                        dstSlices.forward(t, slices, scale);
                        dstSlices.forward(t, 2 * slices, scale);
                        dstSlices.forward(t, 3 * slices, scale);
                        for (int s = 0; s < slices; s++) {
                            idx1 = s * sliceStride + idx0 + c;
                            idx2 = slices + s;
                            a[idx1] = t[s];
                            a[idx1 + 1] = t[idx2];
                            a[idx1 + 2] = t[idx2 + slices];
                            a[idx1 + 3] = t[idx2 + 2 * slices];
                        }
                    }
                }
            } else if (columns == 2) {
                for (int r = 0; r < rows; r++) {
                    idx0 = r * rowStride;
                    for (int s = 0; s < slices; s++) {
                        idx1 = s * sliceStride + idx0;
                        t[s] = a[idx1];
                        t[slices + s] = a[idx1 + 1];
                    }
                    dstSlices.forward(t, 0, scale);
                    dstSlices.forward(t, slices, scale);
                    for (int s = 0; s < slices; s++) {
                        idx1 = s * sliceStride + idx0;
                        a[idx1] = t[s];
                        a[idx1 + 1] = t[slices + s];
                    }
                }
            }
        } else if (columns > 2) {
            for (int r = 0; r < rows; r++) {
                idx0 = r * rowStride;
                for (int c = 0; c < columns; c += 4) {
                    for (int s = 0; s < slices; s++) {
                        idx1 = s * sliceStride + idx0 + c;
                        idx2 = slices + s;
                        t[s] = a[idx1];
                        t[idx2] = a[idx1 + 1];
                        t[idx2 + slices] = a[idx1 + 2];
                        t[idx2 + 2 * slices] = a[idx1 + 3];
                    }
                    dstSlices.inverse(t, 0, scale);
                    dstSlices.inverse(t, slices, scale);
                    dstSlices.inverse(t, 2 * slices, scale);
                    dstSlices.inverse(t, 3 * slices, scale);

                    for (int s = 0; s < slices; s++) {
                        idx1 = s * sliceStride + idx0 + c;
                        idx2 = slices + s;
                        a[idx1] = t[s];
                        a[idx1 + 1] = t[idx2];
                        a[idx1 + 2] = t[idx2 + slices];
                        a[idx1 + 3] = t[idx2 + 2 * slices];
                    }
                }
            }
        } else if (columns == 2) {
            for (int r = 0; r < rows; r++) {
                idx0 = r * rowStride;
                for (int s = 0; s < slices; s++) {
                    idx1 = s * sliceStride + idx0;
                    t[s] = a[idx1];
                    t[slices + s] = a[idx1 + 1];
                }
                dstSlices.inverse(t, 0, scale);
                dstSlices.inverse(t, slices, scale);
                for (int s = 0; s < slices; s++) {
                    idx1 = s * sliceStride + idx0;
                    a[idx1] = t[s];
                    a[idx1 + 1] = t[slices + s];
                }
            }
        }
    }

    private void ddxt3db_sub(int isgn, DoubleLargeArray a, boolean scale)
    {
        long idx0, idx1, idx2;
        long nt = 4 * slicesl;
        if (columnsl == 2) {
            nt >>= 1;
        }
        DoubleLargeArray t = new DoubleLargeArray(nt);
        if (isgn == -1) {
            if (columnsl > 2) {
                for (long r = 0; r < rowsl; r++) {
                    idx0 = r * rowStridel;
                    for (long c = 0; c < columnsl; c += 4) {
                        for (long s = 0; s < slicesl; s++) {
                            idx1 = s * sliceStridel + idx0 + c;
                            idx2 = slicesl + s;
                            t.setDouble(s, a.getDouble(idx1));
                            t.setDouble(idx2, a.getDouble(idx1 + 1));
                            t.setDouble(idx2 + slicesl, a.getDouble(idx1 + 2));
                            t.setDouble(idx2 + 2 * slicesl, a.getDouble(idx1 + 3));
                        }
                        dstSlices.forward(t, 0, scale);
                        dstSlices.forward(t, slicesl, scale);
                        dstSlices.forward(t, 2 * slicesl, scale);
                        dstSlices.forward(t, 3 * slicesl, scale);
                        for (long s = 0; s < slicesl; s++) {
                            idx1 = s * sliceStridel + idx0 + c;
                            idx2 = slicesl + s;
                            a.setDouble(idx1, t.getDouble(s));
                            a.setDouble(idx1 + 1, t.getDouble(idx2));
                            a.setDouble(idx1 + 2, t.getDouble(idx2 + slicesl));
                            a.setDouble(idx1 + 3, t.getDouble(idx2 + 2 * slicesl));
                        }
                    }
                }
            } else if (columnsl == 2) {
                for (long r = 0; r < rowsl; r++) {
                    idx0 = r * rowStridel;
                    for (long s = 0; s < slicesl; s++) {
                        idx1 = s * sliceStridel + idx0;
                        t.setDouble(s, a.getDouble(idx1));
                        t.setDouble(slicesl + s, a.getDouble(idx1 + 1));
                    }
                    dstSlices.forward(t, 0, scale);
                    dstSlices.forward(t, slicesl, scale);
                    for (long s = 0; s < slicesl; s++) {
                        idx1 = s * sliceStridel + idx0;
                        a.setDouble(idx1, t.getDouble(s));
                        a.setDouble(idx1 + 1, t.getDouble(slicesl + s));
                    }
                }
            }
        } else if (columnsl > 2) {
            for (long r = 0; r < rowsl; r++) {
                idx0 = r * rowStridel;
                for (long c = 0; c < columnsl; c += 4) {
                    for (long s = 0; s < slicesl; s++) {
                        idx1 = s * sliceStridel + idx0 + c;
                        idx2 = slicesl + s;
                        t.setDouble(s, a.getDouble(idx1));
                        t.setDouble(idx2, a.getDouble(idx1 + 1));
                        t.setDouble(idx2 + slicesl, a.getDouble(idx1 + 2));
                        t.setDouble(idx2 + 2 * slicesl, a.getDouble(idx1 + 3));
                    }
                    dstSlices.inverse(t, 0, scale);
                    dstSlices.inverse(t, slicesl, scale);
                    dstSlices.inverse(t, 2 * slicesl, scale);
                    dstSlices.inverse(t, 3 * slicesl, scale);

                    for (long s = 0; s < slicesl; s++) {
                        idx1 = s * sliceStridel + idx0 + c;
                        idx2 = slicesl + s;
                        a.setDouble(idx1, t.getDouble(s));
                        a.setDouble(idx1 + 1, t.getDouble(idx2));
                        a.setDouble(idx1 + 2, t.getDouble(idx2 + slicesl));
                        a.setDouble(idx1 + 3, t.getDouble(idx2 + 2 * slicesl));
                    }
                }
            }
        } else if (columnsl == 2) {
            for (long r = 0; r < rowsl; r++) {
                idx0 = r * rowStridel;
                for (long s = 0; s < slicesl; s++) {
                    idx1 = s * sliceStridel + idx0;
                    t.setDouble(s, a.getDouble(idx1));
                    t.setDouble(slicesl + s, a.getDouble(idx1 + 1));
                }
                dstSlices.inverse(t, 0, scale);
                dstSlices.inverse(t, slicesl, scale);
                for (long s = 0; s < slicesl; s++) {
                    idx1 = s * sliceStridel + idx0;
                    a.setDouble(idx1, t.getDouble(s));
                    a.setDouble(idx1 + 1, t.getDouble(slicesl + s));
                }
            }
        }
    }

    private void ddxt3db_sub(int isgn, double[][][] a, boolean scale)
    {
        int idx2;
        int nt = 4 * slices;
        if (columns == 2) {
            nt >>= 1;
        }
        double[] t = new double[nt];

        if (isgn == -1) {
            if (columns > 2) {
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < columns; c += 4) {
                        for (int s = 0; s < slices; s++) {
                            idx2 = slices + s;
                            t[s] = a[s][r][c];
                            t[idx2] = a[s][r][c + 1];
                            t[idx2 + slices] = a[s][r][c + 2];
                            t[idx2 + 2 * slices] = a[s][r][c + 3];
                        }
                        dstSlices.forward(t, 0, scale);
                        dstSlices.forward(t, slices, scale);
                        dstSlices.forward(t, 2 * slices, scale);
                        dstSlices.forward(t, 3 * slices, scale);
                        for (int s = 0; s < slices; s++) {
                            idx2 = slices + s;
                            a[s][r][c] = t[s];
                            a[s][r][c + 1] = t[idx2];
                            a[s][r][c + 2] = t[idx2 + slices];
                            a[s][r][c + 3] = t[idx2 + 2 * slices];
                        }
                    }
                }
            } else if (columns == 2) {
                for (int r = 0; r < rows; r++) {
                    for (int s = 0; s < slices; s++) {
                        t[s] = a[s][r][0];
                        t[slices + s] = a[s][r][1];
                    }
                    dstSlices.forward(t, 0, scale);
                    dstSlices.forward(t, slices, scale);
                    for (int s = 0; s < slices; s++) {
                        a[s][r][0] = t[s];
                        a[s][r][1] = t[slices + s];
                    }
                }
            }
        } else if (columns > 2) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c += 4) {
                    for (int s = 0; s < slices; s++) {
                        idx2 = slices + s;
                        t[s] = a[s][r][c];
                        t[idx2] = a[s][r][c + 1];
                        t[idx2 + slices] = a[s][r][c + 2];
                        t[idx2 + 2 * slices] = a[s][r][c + 3];
                    }
                    dstSlices.inverse(t, 0, scale);
                    dstSlices.inverse(t, slices, scale);
                    dstSlices.inverse(t, 2 * slices, scale);
                    dstSlices.inverse(t, 3 * slices, scale);

                    for (int s = 0; s < slices; s++) {
                        idx2 = slices + s;
                        a[s][r][c] = t[s];
                        a[s][r][c + 1] = t[idx2];
                        a[s][r][c + 2] = t[idx2 + slices];
                        a[s][r][c + 3] = t[idx2 + 2 * slices];
                    }
                }
            }
        } else if (columns == 2) {
            for (int r = 0; r < rows; r++) {
                for (int s = 0; s < slices; s++) {
                    t[s] = a[s][r][0];
                    t[slices + s] = a[s][r][1];
                }
                dstSlices.inverse(t, 0, scale);
                dstSlices.inverse(t, slices, scale);
                for (int s = 0; s < slices; s++) {
                    a[s][r][0] = t[s];
                    a[s][r][1] = t[slices + s];
                }
            }
        }
    }

    private void ddxt3da_subth(final int isgn, final double[] a, final boolean scale)
    {
        final int nthreads = ConcurrencyUtils.getNumberOfThreads() > slices ? slices : ConcurrencyUtils.getNumberOfThreads();

        int nt = 4 * rows;
        if (columns == 2) {
            nt >>= 1;
        }
        final int ntf = nt;
        Future<?>[] futures = new Future[nthreads];

        for (int i = 0; i < nthreads; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {

                public void run()
                {
                    int idx0, idx1, idx2;
                    double[] t = new double[ntf];
                    if (isgn == -1) {
                        for (int s = n0; s < slices; s += nthreads) {
                            idx0 = s * sliceStride;
                            for (int r = 0; r < rows; r++) {
                                dstColumns.forward(a, idx0 + r * rowStride, scale);
                            }
                            if (columns > 2) {
                                for (int c = 0; c < columns; c += 4) {
                                    for (int r = 0; r < rows; r++) {
                                        idx1 = idx0 + r * rowStride + c;
                                        idx2 = rows + r;
                                        t[r] = a[idx1];
                                        t[idx2] = a[idx1 + 1];
                                        t[idx2 + rows] = a[idx1 + 2];
                                        t[idx2 + 2 * rows] = a[idx1 + 3];
                                    }
                                    dstRows.forward(t, 0, scale);
                                    dstRows.forward(t, rows, scale);
                                    dstRows.forward(t, 2 * rows, scale);
                                    dstRows.forward(t, 3 * rows, scale);
                                    for (int r = 0; r < rows; r++) {
                                        idx1 = idx0 + r * rowStride + c;
                                        idx2 = rows + r;
                                        a[idx1] = t[r];
                                        a[idx1 + 1] = t[idx2];
                                        a[idx1 + 2] = t[idx2 + rows];
                                        a[idx1 + 3] = t[idx2 + 2 * rows];
                                    }
                                }
                            } else if (columns == 2) {
                                for (int r = 0; r < rows; r++) {
                                    idx1 = idx0 + r * rowStride;
                                    t[r] = a[idx1];
                                    t[rows + r] = a[idx1 + 1];
                                }
                                dstRows.forward(t, 0, scale);
                                dstRows.forward(t, rows, scale);
                                for (int r = 0; r < rows; r++) {
                                    idx1 = idx0 + r * rowStride;
                                    a[idx1] = t[r];
                                    a[idx1 + 1] = t[rows + r];
                                }
                            }
                        }
                    } else {
                        for (int s = n0; s < slices; s += nthreads) {
                            idx0 = s * sliceStride;
                            for (int r = 0; r < rows; r++) {
                                dstColumns.inverse(a, idx0 + r * rowStride, scale);
                            }
                            if (columns > 2) {
                                for (int c = 0; c < columns; c += 4) {
                                    for (int r = 0; r < rows; r++) {
                                        idx1 = idx0 + r * rowStride + c;
                                        idx2 = rows + r;
                                        t[r] = a[idx1];
                                        t[idx2] = a[idx1 + 1];
                                        t[idx2 + rows] = a[idx1 + 2];
                                        t[idx2 + 2 * rows] = a[idx1 + 3];
                                    }
                                    dstRows.inverse(t, 0, scale);
                                    dstRows.inverse(t, rows, scale);
                                    dstRows.inverse(t, 2 * rows, scale);
                                    dstRows.inverse(t, 3 * rows, scale);
                                    for (int r = 0; r < rows; r++) {
                                        idx1 = idx0 + r * rowStride + c;
                                        idx2 = rows + r;
                                        a[idx1] = t[r];
                                        a[idx1 + 1] = t[idx2];
                                        a[idx1 + 2] = t[idx2 + rows];
                                        a[idx1 + 3] = t[idx2 + 2 * rows];
                                    }
                                }
                            } else if (columns == 2) {
                                for (int r = 0; r < rows; r++) {
                                    idx1 = idx0 + r * rowStride;
                                    t[r] = a[idx1];
                                    t[rows + r] = a[idx1 + 1];
                                }
                                dstRows.inverse(t, 0, scale);
                                dstRows.inverse(t, rows, scale);
                                for (int r = 0; r < rows; r++) {
                                    idx1 = idx0 + r * rowStride;
                                    a[idx1] = t[r];
                                    a[idx1 + 1] = t[rows + r];
                                }
                            }
                        }
                    }
                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void ddxt3da_subth(final int isgn, final DoubleLargeArray a, final boolean scale)
    {
        final int nthreads = (int) (ConcurrencyUtils.getNumberOfThreads() > slicesl ? slicesl : ConcurrencyUtils.getNumberOfThreads());

        long nt = 4 * rowsl;
        if (columnsl == 2) {
            nt >>= 1l;
        }
        final long ntf = nt;
        Future<?>[] futures = new Future[nthreads];

        for (int i = 0; i < nthreads; i++) {
            final long n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {

                public void run()
                {
                    long idx0, idx1, idx2;
                    DoubleLargeArray t = new DoubleLargeArray(ntf);
                    if (isgn == -1) {
                        for (long s = n0; s < slicesl; s += nthreads) {
                            idx0 = s * sliceStridel;
                            for (long r = 0; r < rowsl; r++) {
                                dstColumns.forward(a, idx0 + r * rowStridel, scale);
                            }
                            if (columnsl > 2) {
                                for (long c = 0; c < columnsl; c += 4) {
                                    for (long r = 0; r < rowsl; r++) {
                                        idx1 = idx0 + r * rowStridel + c;
                                        idx2 = rowsl + r;
                                        t.setDouble(r, a.getDouble(idx1));
                                        t.setDouble(idx2, a.getDouble(idx1 + 1));
                                        t.setDouble(idx2 + rowsl, a.getDouble(idx1 + 2));
                                        t.setDouble(idx2 + 2 * rowsl, a.getDouble(idx1 + 3));
                                    }
                                    dstRows.forward(t, 0, scale);
                                    dstRows.forward(t, rowsl, scale);
                                    dstRows.forward(t, 2 * rowsl, scale);
                                    dstRows.forward(t, 3 * rowsl, scale);
                                    for (long r = 0; r < rowsl; r++) {
                                        idx1 = idx0 + r * rowStridel + c;
                                        idx2 = rowsl + r;
                                        a.setDouble(idx1, t.getDouble(r));
                                        a.setDouble(idx1 + 1, t.getDouble(idx2));
                                        a.setDouble(idx1 + 2, t.getDouble(idx2 + rowsl));
                                        a.setDouble(idx1 + 3, t.getDouble(idx2 + 2 * rowsl));
                                    }
                                }
                            } else if (columnsl == 2) {
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStridel;
                                    t.setDouble(r, a.getDouble(idx1));
                                    t.setDouble(rowsl + r, a.getDouble(idx1 + 1));
                                }
                                dstRows.forward(t, 0, scale);
                                dstRows.forward(t, rowsl, scale);
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStridel;
                                    a.setDouble(idx1, t.getDouble(r));
                                    a.setDouble(idx1 + 1, t.getDouble(rowsl + r));
                                }
                            }
                        }
                    } else {
                        for (long s = n0; s < slicesl; s += nthreads) {
                            idx0 = s * sliceStridel;
                            for (long r = 0; r < rowsl; r++) {
                                dstColumns.inverse(a, idx0 + r * rowStridel, scale);
                            }
                            if (columnsl > 2) {
                                for (long c = 0; c < columnsl; c += 4) {
                                    for (long r = 0; r < rowsl; r++) {
                                        idx1 = idx0 + r * rowStridel + c;
                                        idx2 = rowsl + r;
                                        t.setDouble(r, a.getDouble(idx1));
                                        t.setDouble(idx2, a.getDouble(idx1 + 1));
                                        t.setDouble(idx2 + rowsl, a.getDouble(idx1 + 2));
                                        t.setDouble(idx2 + 2 * rowsl, a.getDouble(idx1 + 3));
                                    }
                                    dstRows.inverse(t, 0, scale);
                                    dstRows.inverse(t, rowsl, scale);
                                    dstRows.inverse(t, 2 * rowsl, scale);
                                    dstRows.inverse(t, 3 * rowsl, scale);
                                    for (long r = 0; r < rowsl; r++) {
                                        idx1 = idx0 + r * rowStridel + c;
                                        idx2 = rowsl + r;
                                        a.setDouble(idx1, t.getDouble(r));
                                        a.setDouble(idx1 + 1, t.getDouble(idx2));
                                        a.setDouble(idx1 + 2, t.getDouble(idx2 + rowsl));
                                        a.setDouble(idx1 + 3, t.getDouble(idx2 + 2 * rowsl));
                                    }
                                }
                            } else if (columnsl == 2) {
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStridel;
                                    t.setDouble(r, a.getDouble(idx1));
                                    t.setDouble(rowsl + r, a.getDouble(idx1 + 1));
                                }
                                dstRows.inverse(t, 0, scale);
                                dstRows.inverse(t, rowsl, scale);
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStridel;
                                    a.setDouble(idx1, t.getDouble(r));
                                    a.setDouble(idx1 + 1, t.getDouble(rowsl + r));
                                }
                            }
                        }
                    }
                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void ddxt3da_subth(final int isgn, final double[][][] a, final boolean scale)
    {
        final int nthreads = ConcurrencyUtils.getNumberOfThreads() > slices ? slices : ConcurrencyUtils.getNumberOfThreads();
        int nt = 4 * rows;
        if (columns == 2) {
            nt >>= 1;
        }
        final int ntf = nt;
        Future<?>[] futures = new Future[nthreads];

        for (int i = 0; i < nthreads; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {

                public void run()
                {
                    int idx2;
                    double[] t = new double[ntf];
                    if (isgn == -1) {
                        for (int s = n0; s < slices; s += nthreads) {
                            for (int r = 0; r < rows; r++) {
                                dstColumns.forward(a[s][r], scale);
                            }
                            if (columns > 2) {
                                for (int c = 0; c < columns; c += 4) {
                                    for (int r = 0; r < rows; r++) {
                                        idx2 = rows + r;
                                        t[r] = a[s][r][c];
                                        t[idx2] = a[s][r][c + 1];
                                        t[idx2 + rows] = a[s][r][c + 2];
                                        t[idx2 + 2 * rows] = a[s][r][c + 3];
                                    }
                                    dstRows.forward(t, 0, scale);
                                    dstRows.forward(t, rows, scale);
                                    dstRows.forward(t, 2 * rows, scale);
                                    dstRows.forward(t, 3 * rows, scale);
                                    for (int r = 0; r < rows; r++) {
                                        idx2 = rows + r;
                                        a[s][r][c] = t[r];
                                        a[s][r][c + 1] = t[idx2];
                                        a[s][r][c + 2] = t[idx2 + rows];
                                        a[s][r][c + 3] = t[idx2 + 2 * rows];
                                    }
                                }
                            } else if (columns == 2) {
                                for (int r = 0; r < rows; r++) {
                                    t[r] = a[s][r][0];
                                    t[rows + r] = a[s][r][1];
                                }
                                dstRows.forward(t, 0, scale);
                                dstRows.forward(t, rows, scale);
                                for (int r = 0; r < rows; r++) {
                                    a[s][r][0] = t[r];
                                    a[s][r][1] = t[rows + r];
                                }
                            }
                        }
                    } else {
                        for (int s = n0; s < slices; s += nthreads) {
                            for (int r = 0; r < rows; r++) {
                                dstColumns.inverse(a[s][r], scale);
                            }
                            if (columns > 2) {
                                for (int c = 0; c < columns; c += 4) {
                                    for (int r = 0; r < rows; r++) {
                                        idx2 = rows + r;
                                        t[r] = a[s][r][c];
                                        t[idx2] = a[s][r][c + 1];
                                        t[idx2 + rows] = a[s][r][c + 2];
                                        t[idx2 + 2 * rows] = a[s][r][c + 3];
                                    }
                                    dstRows.inverse(t, 0, scale);
                                    dstRows.inverse(t, rows, scale);
                                    dstRows.inverse(t, 2 * rows, scale);
                                    dstRows.inverse(t, 3 * rows, scale);
                                    for (int r = 0; r < rows; r++) {
                                        idx2 = rows + r;
                                        a[s][r][c] = t[r];
                                        a[s][r][c + 1] = t[idx2];
                                        a[s][r][c + 2] = t[idx2 + rows];
                                        a[s][r][c + 3] = t[idx2 + 2 * rows];
                                    }
                                }
                            } else if (columns == 2) {
                                for (int r = 0; r < rows; r++) {
                                    t[r] = a[s][r][0];
                                    t[rows + r] = a[s][r][1];
                                }
                                dstRows.inverse(t, 0, scale);
                                dstRows.inverse(t, rows, scale);
                                for (int r = 0; r < rows; r++) {
                                    a[s][r][0] = t[r];
                                    a[s][r][1] = t[rows + r];
                                }
                            }
                        }
                    }
                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void ddxt3db_subth(final int isgn, final double[] a, final boolean scale)
    {
        final int nthreads = ConcurrencyUtils.getNumberOfThreads() > rows ? rows : ConcurrencyUtils.getNumberOfThreads();
        int nt = 4 * slices;
        if (columns == 2) {
            nt >>= 1;
        }
        final int ntf = nt;
        Future<?>[] futures = new Future[nthreads];

        for (int i = 0; i < nthreads; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {

                public void run()
                {
                    int idx0, idx1, idx2;
                    double[] t = new double[ntf];
                    if (isgn == -1) {
                        if (columns > 2) {
                            for (int r = n0; r < rows; r += nthreads) {
                                idx0 = r * rowStride;
                                for (int c = 0; c < columns; c += 4) {
                                    for (int s = 0; s < slices; s++) {
                                        idx1 = s * sliceStride + idx0 + c;
                                        idx2 = slices + s;
                                        t[s] = a[idx1];
                                        t[idx2] = a[idx1 + 1];
                                        t[idx2 + slices] = a[idx1 + 2];
                                        t[idx2 + 2 * slices] = a[idx1 + 3];
                                    }
                                    dstSlices.forward(t, 0, scale);
                                    dstSlices.forward(t, slices, scale);
                                    dstSlices.forward(t, 2 * slices, scale);
                                    dstSlices.forward(t, 3 * slices, scale);
                                    for (int s = 0; s < slices; s++) {
                                        idx1 = s * sliceStride + idx0 + c;
                                        idx2 = slices + s;
                                        a[idx1] = t[s];
                                        a[idx1 + 1] = t[idx2];
                                        a[idx1 + 2] = t[idx2 + slices];
                                        a[idx1 + 3] = t[idx2 + 2 * slices];
                                    }
                                }
                            }
                        } else if (columns == 2) {
                            for (int r = n0; r < rows; r += nthreads) {
                                idx0 = r * rowStride;
                                for (int s = 0; s < slices; s++) {
                                    idx1 = s * sliceStride + idx0;
                                    t[s] = a[idx1];
                                    t[slices + s] = a[idx1 + 1];
                                }
                                dstSlices.forward(t, 0, scale);
                                dstSlices.forward(t, slices, scale);
                                for (int s = 0; s < slices; s++) {
                                    idx1 = s * sliceStride + idx0;
                                    a[idx1] = t[s];
                                    a[idx1 + 1] = t[slices + s];
                                }
                            }
                        }
                    } else if (columns > 2) {
                        for (int r = n0; r < rows; r += nthreads) {
                            idx0 = r * rowStride;
                            for (int c = 0; c < columns; c += 4) {
                                for (int s = 0; s < slices; s++) {
                                    idx1 = s * sliceStride + idx0 + c;
                                    idx2 = slices + s;
                                    t[s] = a[idx1];
                                    t[idx2] = a[idx1 + 1];
                                    t[idx2 + slices] = a[idx1 + 2];
                                    t[idx2 + 2 * slices] = a[idx1 + 3];
                                }
                                dstSlices.inverse(t, 0, scale);
                                dstSlices.inverse(t, slices, scale);
                                dstSlices.inverse(t, 2 * slices, scale);
                                dstSlices.inverse(t, 3 * slices, scale);
                                for (int s = 0; s < slices; s++) {
                                    idx1 = s * sliceStride + idx0 + c;
                                    idx2 = slices + s;
                                    a[idx1] = t[s];
                                    a[idx1 + 1] = t[idx2];
                                    a[idx1 + 2] = t[idx2 + slices];
                                    a[idx1 + 3] = t[idx2 + 2 * slices];
                                }
                            }
                        }
                    } else if (columns == 2) {
                        for (int r = n0; r < rows; r += nthreads) {
                            idx0 = r * rowStride;
                            for (int s = 0; s < slices; s++) {
                                idx1 = s * sliceStride + idx0;
                                t[s] = a[idx1];
                                t[slices + s] = a[idx1 + 1];
                            }
                            dstSlices.inverse(t, 0, scale);
                            dstSlices.inverse(t, slices, scale);

                            for (int s = 0; s < slices; s++) {
                                idx1 = s * sliceStride + idx0;
                                a[idx1] = t[s];
                                a[idx1 + 1] = t[slices + s];
                            }
                        }
                    }
                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void ddxt3db_subth(final int isgn, final DoubleLargeArray a, final boolean scale)
    {
        final int nthreads = (int) (ConcurrencyUtils.getNumberOfThreads() > rowsl ? rowsl : ConcurrencyUtils.getNumberOfThreads());
        long nt = 4 * slicesl;
        if (columnsl == 2) {
            nt >>= 1;
        }
        final long ntf = nt;
        Future<?>[] futures = new Future[nthreads];

        for (int i = 0; i < nthreads; i++) {
            final long n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {

                public void run()
                {
                    long idx0, idx1, idx2;
                    DoubleLargeArray t = new DoubleLargeArray(ntf);
                    if (isgn == -1) {
                        if (columnsl > 2) {
                            for (long r = n0; r < rowsl; r += nthreads) {
                                idx0 = r * rowStridel;
                                for (long c = 0; c < columnsl; c += 4) {
                                    for (long s = 0; s < slicesl; s++) {
                                        idx1 = s * sliceStridel + idx0 + c;
                                        idx2 = slicesl + s;
                                        t.setDouble(s, a.getDouble(idx1));
                                        t.setDouble(idx2, a.getDouble(idx1 + 1));
                                        t.setDouble(idx2 + slicesl, a.getDouble(idx1 + 2));
                                        t.setDouble(idx2 + 2 * slicesl, a.getDouble(idx1 + 3));
                                    }
                                    dstSlices.forward(t, 0, scale);
                                    dstSlices.forward(t, slicesl, scale);
                                    dstSlices.forward(t, 2 * slicesl, scale);
                                    dstSlices.forward(t, 3 * slicesl, scale);
                                    for (long s = 0; s < slicesl; s++) {
                                        idx1 = s * sliceStridel + idx0 + c;
                                        idx2 = slicesl + s;
                                        a.setDouble(idx1, t.getDouble(s));
                                        a.setDouble(idx1 + 1, t.getDouble(idx2));
                                        a.setDouble(idx1 + 2, t.getDouble(idx2 + slicesl));
                                        a.setDouble(idx1 + 3, t.getDouble(idx2 + 2 * slicesl));
                                    }
                                }
                            }
                        } else if (columnsl == 2) {
                            for (long r = n0; r < rowsl; r += nthreads) {
                                idx0 = r * rowStridel;
                                for (long s = 0; s < slicesl; s++) {
                                    idx1 = s * sliceStridel + idx0;
                                    t.setDouble(s, a.getDouble(idx1));
                                    t.setDouble(slicesl + s, a.getDouble(idx1 + 1));
                                }
                                dstSlices.forward(t, 0, scale);
                                dstSlices.forward(t, slicesl, scale);
                                for (long s = 0; s < slicesl; s++) {
                                    idx1 = s * sliceStridel + idx0;
                                    a.setDouble(idx1, t.getDouble(s));
                                    a.setDouble(idx1 + 1, t.getDouble(slicesl + s));
                                }
                            }
                        }
                    } else if (columnsl > 2) {
                        for (long r = n0; r < rowsl; r += nthreads) {
                            idx0 = r * rowStridel;
                            for (long c = 0; c < columnsl; c += 4) {
                                for (long s = 0; s < slicesl; s++) {
                                    idx1 = s * sliceStridel + idx0 + c;
                                    idx2 = slicesl + s;
                                    t.setDouble(s, a.getDouble(idx1));
                                    t.setDouble(idx2, a.getDouble(idx1 + 1));
                                    t.setDouble(idx2 + slicesl, a.getDouble(idx1 + 2));
                                    t.setDouble(idx2 + 2 * slicesl, a.getDouble(idx1 + 3));
                                }
                                dstSlices.inverse(t, 0, scale);
                                dstSlices.inverse(t, slicesl, scale);
                                dstSlices.inverse(t, 2 * slicesl, scale);
                                dstSlices.inverse(t, 3 * slicesl, scale);
                                for (long s = 0; s < slicesl; s++) {
                                    idx1 = s * sliceStridel + idx0 + c;
                                    idx2 = slicesl + s;
                                    a.setDouble(idx1, t.getDouble(s));
                                    a.setDouble(idx1 + 1, t.getDouble(idx2));
                                    a.setDouble(idx1 + 2, t.getDouble(idx2 + slicesl));
                                    a.setDouble(idx1 + 3, t.getDouble(idx2 + 2 * slicesl));
                                }
                            }
                        }
                    } else if (columnsl == 2) {
                        for (long r = n0; r < rowsl; r += nthreads) {
                            idx0 = r * rowStridel;
                            for (long s = 0; s < slicesl; s++) {
                                idx1 = s * sliceStridel + idx0;
                                t.setDouble(s, a.getDouble(idx1));
                                t.setDouble(slicesl + s, a.getDouble(idx1 + 1));
                            }
                            dstSlices.inverse(t, 0, scale);
                            dstSlices.inverse(t, slicesl, scale);

                            for (long s = 0; s < slicesl; s++) {
                                idx1 = s * sliceStridel + idx0;
                                a.setDouble(idx1, t.getDouble(s));
                                a.setDouble(idx1 + 1, t.getDouble(slicesl + s));
                            }
                        }
                    }
                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void ddxt3db_subth(final int isgn, final double[][][] a, final boolean scale)
    {
        final int nthreads = ConcurrencyUtils.getNumberOfThreads() > rows ? rows : ConcurrencyUtils.getNumberOfThreads();
        int nt = 4 * slices;
        if (columns == 2) {
            nt >>= 1;
        }
        final int ntf = nt;
        Future<?>[] futures = new Future[nthreads];

        for (int i = 0; i < nthreads; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {

                public void run()
                {
                    int idx2;
                    double[] t = new double[ntf];
                    if (isgn == -1) {
                        if (columns > 2) {
                            for (int r = n0; r < rows; r += nthreads) {
                                for (int c = 0; c < columns; c += 4) {
                                    for (int s = 0; s < slices; s++) {
                                        idx2 = slices + s;
                                        t[s] = a[s][r][c];
                                        t[idx2] = a[s][r][c + 1];
                                        t[idx2 + slices] = a[s][r][c + 2];
                                        t[idx2 + 2 * slices] = a[s][r][c + 3];
                                    }
                                    dstSlices.forward(t, 0, scale);
                                    dstSlices.forward(t, slices, scale);
                                    dstSlices.forward(t, 2 * slices, scale);
                                    dstSlices.forward(t, 3 * slices, scale);
                                    for (int s = 0; s < slices; s++) {
                                        idx2 = slices + s;
                                        a[s][r][c] = t[s];
                                        a[s][r][c + 1] = t[idx2];
                                        a[s][r][c + 2] = t[idx2 + slices];
                                        a[s][r][c + 3] = t[idx2 + 2 * slices];
                                    }
                                }
                            }
                        } else if (columns == 2) {
                            for (int r = n0; r < rows; r += nthreads) {
                                for (int s = 0; s < slices; s++) {
                                    t[s] = a[s][r][0];
                                    t[slices + s] = a[s][r][1];
                                }
                                dstSlices.forward(t, 0, scale);
                                dstSlices.forward(t, slices, scale);
                                for (int s = 0; s < slices; s++) {
                                    a[s][r][0] = t[s];
                                    a[s][r][1] = t[slices + s];
                                }
                            }
                        }
                    } else if (columns > 2) {
                        for (int r = n0; r < rows; r += nthreads) {
                            for (int c = 0; c < columns; c += 4) {
                                for (int s = 0; s < slices; s++) {
                                    idx2 = slices + s;
                                    t[s] = a[s][r][c];
                                    t[idx2] = a[s][r][c + 1];
                                    t[idx2 + slices] = a[s][r][c + 2];
                                    t[idx2 + 2 * slices] = a[s][r][c + 3];
                                }
                                dstSlices.inverse(t, 0, scale);
                                dstSlices.inverse(t, slices, scale);
                                dstSlices.inverse(t, 2 * slices, scale);
                                dstSlices.inverse(t, 3 * slices, scale);
                                for (int s = 0; s < slices; s++) {
                                    idx2 = slices + s;
                                    a[s][r][c] = t[s];
                                    a[s][r][c + 1] = t[idx2];
                                    a[s][r][c + 2] = t[idx2 + slices];
                                    a[s][r][c + 3] = t[idx2 + 2 * slices];
                                }
                            }
                        }
                    } else if (columns == 2) {
                        for (int r = n0; r < rows; r += nthreads) {
                            for (int s = 0; s < slices; s++) {
                                t[s] = a[s][r][0];
                                t[slices + s] = a[s][r][1];
                            }
                            dstSlices.inverse(t, 0, scale);
                            dstSlices.inverse(t, slices, scale);

                            for (int s = 0; s < slices; s++) {
                                a[s][r][0] = t[s];
                                a[s][r][1] = t[slices + s];
                            }
                        }
                    }
                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(DoubleDST_3D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
