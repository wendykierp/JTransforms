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

/**
 * Computes 3D Discrete Cosine Transform (DCT) of single precision data. The
 * sizes of all three dimensions can be arbitrary numbers. This is a parallel
 * implementation of split-radix and mixed-radix algorithms optimized for SMP
 * systems. <br>
 * <br>
 * Part of the code is derived from General Purpose FFT Package written by
 * Takuya Ooura (http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html)
 *  
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class FloatDCT_3D
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

    private FloatDCT_1D dctSlices, dctRows, dctColumns;

    private boolean isPowerOfTwo = false;

    private boolean useThreads = false;

    /**
     * Creates new instance of FloatDCT_3D.
     *  
     * @param slices  number of slices
     * @param rows    number of rows
     * @param columns number of columns
     */
    public FloatDCT_3D(long slices, long rows, long columns)
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
        dctSlices = new FloatDCT_1D(slices);
        if (slices == rows) {
            dctRows = dctSlices;
        } else {
            dctRows = new FloatDCT_1D(rows);
        }
        if (slices == columns) {
            dctColumns = dctSlices;
        } else if (rows == columns) {
            dctColumns = dctRows;
        } else {
            dctColumns = new FloatDCT_1D(columns);
        }
    }

    /**
     * Computes the 3D forward DCT (DCT-II) leaving the result in <code>a</code>
     * . The data is stored in 1D array addressed in slice-major, then
     * row-major, then column-major, in order of significance, i.e. the element
     * (i,j,k) of 3D array x[slices][rows][columns] is stored in a[i*sliceStride
     * + j*rowStride + k], where sliceStride = rows * columns and rowStride =
     * columns.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void forward(final float[] a, final boolean scale)
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
                                dctColumns.forward(a, idx1 + r * rowStride, scale);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        float[] temp = new float[rows];
                        for (int s = firstSlice; s < lastSlice; s++) {
                            int idx1 = s * sliceStride;
                            for (int c = 0; c < columns; c++) {
                                for (int r = 0; r < rows; r++) {
                                    int idx3 = idx1 + r * rowStride + c;
                                    temp[r] = a[idx3];
                                }
                                dctRows.forward(temp, scale);
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
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            p = rows / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstRow = l * p;
                final int lastRow = (l == (nthreads - 1)) ? rows : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        float[] temp = new float[slices];
                        for (int r = firstRow; r < lastRow; r++) {
                            int idx1 = r * rowStride;
                            for (int c = 0; c < columns; c++) {
                                for (int s = 0; s < slices; s++) {
                                    int idx3 = s * sliceStride + idx1 + c;
                                    temp[s] = a[idx3];
                                }
                                dctSlices.forward(temp, scale);
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
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

        } else {
            for (int s = 0; s < slices; s++) {
                int idx1 = s * sliceStride;
                for (int r = 0; r < rows; r++) {
                    dctColumns.forward(a, idx1 + r * rowStride, scale);
                }
            }
            float[] temp = new float[rows];
            for (int s = 0; s < slices; s++) {
                int idx1 = s * sliceStride;
                for (int c = 0; c < columns; c++) {
                    for (int r = 0; r < rows; r++) {
                        int idx3 = idx1 + r * rowStride + c;
                        temp[r] = a[idx3];
                    }
                    dctRows.forward(temp, scale);
                    for (int r = 0; r < rows; r++) {
                        int idx3 = idx1 + r * rowStride + c;
                        a[idx3] = temp[r];
                    }
                }
            }
            temp = new float[slices];
            for (int r = 0; r < rows; r++) {
                int idx1 = r * rowStride;
                for (int c = 0; c < columns; c++) {
                    for (int s = 0; s < slices; s++) {
                        int idx3 = s * sliceStride + idx1 + c;
                        temp[s] = a[idx3];
                    }
                    dctSlices.forward(temp, scale);
                    for (int s = 0; s < slices; s++) {
                        int idx3 = s * sliceStride + idx1 + c;
                        a[idx3] = temp[s];
                    }
                }
            }
        }
    }

    /**
     * Computes the 3D forward DCT (DCT-II) leaving the result in <code>a</code>
     * . The data is stored in 1D array addressed in slice-major, then
     * row-major, then column-major, in order of significance, i.e. the element
     * (i,j,k) of 3D array x[slices][rows][columns] is stored in a[i*sliceStride
     * + j*rowStride + k], where sliceStride = rows * columns and rowStride =
     * columns.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void forward(final FloatLargeArray a, final boolean scale)
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
                                dctColumns.forward(a, idx1 + r * rowStridel, scale);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int l = 0; l < nthreads; l++) {
                final long firstSlice = l * p;
                final long lastSlice = (l == (nthreads - 1)) ? slicesl : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        FloatLargeArray temp = new FloatLargeArray(rowsl, false);
                        for (long s = firstSlice; s < lastSlice; s++) {
                            long idx1 = s * sliceStridel;
                            for (long c = 0; c < columnsl; c++) {
                                for (long r = 0; r < rowsl; r++) {
                                    long idx3 = idx1 + r * rowStridel + c;
                                    temp.setFloat(r, a.getFloat(idx3));
                                }
                                dctRows.forward(temp, scale);
                                for (long r = 0; r < rowsl; r++) {
                                    long idx3 = idx1 + r * rowStridel + c;
                                    a.setFloat(idx3, temp.getFloat(r));
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            p = rowsl / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final long firstRow = l * p;
                final long lastRow = (l == (nthreads - 1)) ? rowsl : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        FloatLargeArray temp = new FloatLargeArray(slicesl, false);
                        for (long r = firstRow; r < lastRow; r++) {
                            long idx1 = r * rowStridel;
                            for (long c = 0; c < columnsl; c++) {
                                for (long s = 0; s < slicesl; s++) {
                                    long idx3 = s * sliceStridel + idx1 + c;
                                    temp.setFloat(s, a.getFloat(idx3));
                                }
                                dctSlices.forward(temp, scale);
                                for (long s = 0; s < slicesl; s++) {
                                    long idx3 = s * sliceStridel + idx1 + c;
                                    a.setFloat(idx3, temp.getFloat(s));
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

        } else {
            for (long s = 0; s < slicesl; s++) {
                long idx1 = s * sliceStridel;
                for (long r = 0; r < rowsl; r++) {
                    dctColumns.forward(a, idx1 + r * rowStridel, scale);
                }
            }
            FloatLargeArray temp = new FloatLargeArray(rowsl, false);
            for (long s = 0; s < slicesl; s++) {
                long idx1 = s * sliceStridel;
                for (long c = 0; c < columnsl; c++) {
                    for (long r = 0; r < rowsl; r++) {
                        long idx3 = idx1 + r * rowStridel + c;
                        temp.setFloat(r, a.getFloat(idx3));
                    }
                    dctRows.forward(temp, scale);
                    for (long r = 0; r < rowsl; r++) {
                        long idx3 = idx1 + r * rowStridel + c;
                        a.setFloat(idx3, temp.getFloat(r));
                    }
                }
            }
            temp = new FloatLargeArray(slicesl, false);
            for (long r = 0; r < rowsl; r++) {
                long idx1 = r * rowStridel;
                for (long c = 0; c < columnsl; c++) {
                    for (long s = 0; s < slicesl; s++) {
                        long idx3 = s * sliceStridel + idx1 + c;
                        temp.setFloat(s, a.getFloat(idx3));
                    }
                    dctSlices.forward(temp, scale);
                    for (long s = 0; s < slicesl; s++) {
                        long idx3 = s * sliceStridel + idx1 + c;
                        a.setFloat(idx3, temp.getFloat(s));
                    }
                }
            }
        }
    }

    /**
     * Computes the 3D forward DCT (DCT-II) leaving the result in <code>a</code>
     * . The data is stored in 3D array
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void forward(final float[][][] a, final boolean scale)
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
                                dctColumns.forward(a[s][r], scale);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        float[] temp = new float[rows];
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int c = 0; c < columns; c++) {
                                for (int r = 0; r < rows; r++) {
                                    temp[r] = a[s][r][c];
                                }
                                dctRows.forward(temp, scale);
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
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            p = rows / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstRow = l * p;
                final int lastRow = (l == (nthreads - 1)) ? rows : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        float[] temp = new float[slices];
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                for (int s = 0; s < slices; s++) {
                                    temp[s] = a[s][r][c];
                                }
                                dctSlices.forward(temp, scale);
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
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

        } else {
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    dctColumns.forward(a[s][r], scale);
                }
            }
            float[] temp = new float[rows];
            for (int s = 0; s < slices; s++) {
                for (int c = 0; c < columns; c++) {
                    for (int r = 0; r < rows; r++) {
                        temp[r] = a[s][r][c];
                    }
                    dctRows.forward(temp, scale);
                    for (int r = 0; r < rows; r++) {
                        a[s][r][c] = temp[r];
                    }
                }
            }
            temp = new float[slices];
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    for (int s = 0; s < slices; s++) {
                        temp[s] = a[s][r][c];
                    }
                    dctSlices.forward(temp, scale);
                    for (int s = 0; s < slices; s++) {
                        a[s][r][c] = temp[s];
                    }
                }
            }
        }
    }

    /**
     * Computes the 3D inverse DCT (DCT-III) leaving the result in
     * <code>a</code>. The data is stored in 1D array addressed in slice-major,
     * then row-major, then column-major, in order of significance, i.e. the
     * element (i,j,k) of 3D array x[slices][rows][columns] is stored in
     * a[i*sliceStride + j*rowStride + k], where sliceStride = rows * columns
     * and rowStride = columns.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void inverse(final float[] a, final boolean scale)
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
                                dctColumns.inverse(a, idx1 + r * rowStride, scale);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        float[] temp = new float[rows];
                        for (int s = firstSlice; s < lastSlice; s++) {
                            int idx1 = s * sliceStride;
                            for (int c = 0; c < columns; c++) {
                                for (int r = 0; r < rows; r++) {
                                    int idx3 = idx1 + r * rowStride + c;
                                    temp[r] = a[idx3];
                                }
                                dctRows.inverse(temp, scale);
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
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            p = rows / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstRow = l * p;
                final int lastRow = (l == (nthreads - 1)) ? rows : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        float[] temp = new float[slices];
                        for (int r = firstRow; r < lastRow; r++) {
                            int idx1 = r * rowStride;
                            for (int c = 0; c < columns; c++) {
                                for (int s = 0; s < slices; s++) {
                                    int idx3 = s * sliceStride + idx1 + c;
                                    temp[s] = a[idx3];
                                }
                                dctSlices.inverse(temp, scale);
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
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

        } else {
            for (int s = 0; s < slices; s++) {
                int idx1 = s * sliceStride;
                for (int r = 0; r < rows; r++) {
                    dctColumns.inverse(a, idx1 + r * rowStride, scale);
                }
            }
            float[] temp = new float[rows];
            for (int s = 0; s < slices; s++) {
                int idx1 = s * sliceStride;
                for (int c = 0; c < columns; c++) {
                    for (int r = 0; r < rows; r++) {
                        int idx3 = idx1 + r * rowStride + c;
                        temp[r] = a[idx3];
                    }
                    dctRows.inverse(temp, scale);
                    for (int r = 0; r < rows; r++) {
                        int idx3 = idx1 + r * rowStride + c;
                        a[idx3] = temp[r];
                    }
                }
            }
            temp = new float[slices];
            for (int r = 0; r < rows; r++) {
                int idx1 = r * rowStride;
                for (int c = 0; c < columns; c++) {
                    for (int s = 0; s < slices; s++) {
                        int idx3 = s * sliceStride + idx1 + c;
                        temp[s] = a[idx3];
                    }
                    dctSlices.inverse(temp, scale);
                    for (int s = 0; s < slices; s++) {
                        int idx3 = s * sliceStride + idx1 + c;
                        a[idx3] = temp[s];
                    }
                }
            }
        }
    }

    /**
     * Computes the 3D inverse DCT (DCT-III) leaving the result in
     * <code>a</code>. The data is stored in 1D array addressed in slice-major,
     * then row-major, then column-major, in order of significance, i.e. the
     * element (i,j,k) of 3D array x[slices][rows][columns] is stored in
     * a[i*sliceStride + j*rowStride + k], where sliceStride = rows * columns
     * and rowStride = columns.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void inverse(final FloatLargeArray a, final boolean scale)
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
                            long idx1 = s * sliceStridel;
                            for (long r = 0; r < rowsl; r++) {
                                dctColumns.inverse(a, idx1 + r * rowStridel, scale);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int l = 0; l < nthreads; l++) {
                final long firstSlice = l * p;
                final long lastSlice = (l == (nthreads - 1)) ? slicesl : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        FloatLargeArray temp = new FloatLargeArray(rowsl, false);
                        for (long s = firstSlice; s < lastSlice; s++) {
                            long idx1 = s * sliceStridel;
                            for (long c = 0; c < columnsl; c++) {
                                for (long r = 0; r < rowsl; r++) {
                                    long idx3 = idx1 + r * rowStridel + c;
                                    temp.setFloat(r, a.getFloat(idx3));
                                }
                                dctRows.inverse(temp, scale);
                                for (long r = 0; r < rowsl; r++) {
                                    long idx3 = idx1 + r * rowStridel + c;
                                    a.setFloat(idx3, temp.getFloat(r));
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            p = rowsl / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final long firstRow = l * p;
                final long lastRow = (l == (nthreads - 1)) ? rowsl : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        FloatLargeArray temp = new FloatLargeArray(slicesl, false);
                        for (long r = firstRow; r < lastRow; r++) {
                            long idx1 = r * rowStridel;
                            for (long c = 0; c < columnsl; c++) {
                                for (long s = 0; s < slicesl; s++) {
                                    long idx3 = s * sliceStridel + idx1 + c;
                                    temp.setFloat(s, a.getFloat(idx3));
                                }
                                dctSlices.inverse(temp, scale);
                                for (long s = 0; s < slicesl; s++) {
                                    long idx3 = s * sliceStridel + idx1 + c;
                                    a.setFloat(idx3, temp.getFloat(s));
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

        } else {
            for (long s = 0; s < slicesl; s++) {
                long idx1 = s * sliceStridel;
                for (long r = 0; r < rowsl; r++) {
                    dctColumns.inverse(a, idx1 + r * rowStridel, scale);
                }
            }
            FloatLargeArray temp = new FloatLargeArray(rowsl, false);
            for (long s = 0; s < slicesl; s++) {
                long idx1 = s * sliceStridel;
                for (long c = 0; c < columnsl; c++) {
                    for (long r = 0; r < rowsl; r++) {
                        long idx3 = idx1 + r * rowStridel + c;
                        temp.setFloat(r, a.getFloat(idx3));
                    }
                    dctRows.inverse(temp, scale);
                    for (long r = 0; r < rowsl; r++) {
                        long idx3 = idx1 + r * rowStridel + c;
                        a.setFloat(idx3, temp.getFloat(r));
                    }
                }
            }
            temp = new FloatLargeArray(slicesl, false);
            for (long r = 0; r < rowsl; r++) {
                long idx1 = r * rowStridel;
                for (long c = 0; c < columnsl; c++) {
                    for (long s = 0; s < slicesl; s++) {
                        long idx3 = s * sliceStridel + idx1 + c;
                        temp.setFloat(s, a.getFloat(idx3));
                    }
                    dctSlices.inverse(temp, scale);
                    for (long s = 0; s < slicesl; s++) {
                        long idx3 = s * sliceStridel + idx1 + c;
                        a.setFloat(idx3, temp.getFloat(s));
                    }
                }
            }
        }
    }

    /**
     * Computes the 3D inverse DCT (DCT-III) leaving the result in
     * <code>a</code>. The data is stored in 3D array.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void inverse(final float[][][] a, final boolean scale)
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
                                dctColumns.inverse(a[s][r], scale);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        float[] temp = new float[rows];
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int c = 0; c < columns; c++) {
                                for (int r = 0; r < rows; r++) {
                                    temp[r] = a[s][r][c];
                                }
                                dctRows.inverse(temp, scale);
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
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            p = rows / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstRow = l * p;
                final int lastRow = (l == (nthreads - 1)) ? rows : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        float[] temp = new float[slices];
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                for (int s = 0; s < slices; s++) {
                                    temp[s] = a[s][r][c];
                                }
                                dctSlices.inverse(temp, scale);
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
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

        } else {
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    dctColumns.inverse(a[s][r], scale);
                }
            }
            float[] temp = new float[rows];
            for (int s = 0; s < slices; s++) {
                for (int c = 0; c < columns; c++) {
                    for (int r = 0; r < rows; r++) {
                        temp[r] = a[s][r][c];
                    }
                    dctRows.inverse(temp, scale);
                    for (int r = 0; r < rows; r++) {
                        a[s][r][c] = temp[r];
                    }
                }
            }
            temp = new float[slices];
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    for (int s = 0; s < slices; s++) {
                        temp[s] = a[s][r][c];
                    }
                    dctSlices.inverse(temp, scale);
                    for (int s = 0; s < slices; s++) {
                        a[s][r][c] = temp[s];
                    }
                }
            }
        }
    }

    private void ddxt3da_sub(int isgn, float[] a, boolean scale)
    {
        int idx0, idx1, idx2;
        int nt = 4 * rows;
        if (columns == 2) {
            nt >>= 1;
        }
        float[] t = new float[nt];
        if (isgn == -1) {
            for (int s = 0; s < slices; s++) {
                idx0 = s * sliceStride;
                for (int r = 0; r < rows; r++) {
                    dctColumns.forward(a, idx0 + r * rowStride, scale);
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
                        dctRows.forward(t, 0, scale);
                        dctRows.forward(t, rows, scale);
                        dctRows.forward(t, 2 * rows, scale);
                        dctRows.forward(t, 3 * rows, scale);
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
                    dctRows.forward(t, 0, scale);
                    dctRows.forward(t, rows, scale);
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
                    dctColumns.inverse(a, idx0 + r * rowStride, scale);
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
                        dctRows.inverse(t, 0, scale);
                        dctRows.inverse(t, rows, scale);
                        dctRows.inverse(t, 2 * rows, scale);
                        dctRows.inverse(t, 3 * rows, scale);
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
                    dctRows.inverse(t, 0, scale);
                    dctRows.inverse(t, rows, scale);
                    for (int r = 0; r < rows; r++) {
                        idx1 = idx0 + r * rowStride;
                        a[idx1] = t[r];
                        a[idx1 + 1] = t[rows + r];
                    }
                }
            }
        }
    }

    private void ddxt3da_sub(int isgn, FloatLargeArray a, boolean scale)
    {
        long idx0, idx1, idx2;
        long nt = 4 * rowsl;
        if (columnsl == 2) {
            nt >>= 1;
        }
        FloatLargeArray t = new FloatLargeArray(nt);
        if (isgn == -1) {
            for (long s = 0; s < slicesl; s++) {
                idx0 = s * sliceStridel;
                for (long r = 0; r < rowsl; r++) {
                    dctColumns.forward(a, idx0 + r * rowStridel, scale);
                }
                if (columnsl > 2) {
                    for (long c = 0; c < columnsl; c += 4) {
                        for (long r = 0; r < rowsl; r++) {
                            idx1 = idx0 + r * rowStridel + c;
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
                            idx1 = idx0 + r * rowStridel + c;
                            idx2 = rowsl + r;
                            a.setFloat(idx1, t.getFloat(r));
                            a.setFloat(idx1 + 1, t.getFloat(idx2));
                            a.setFloat(idx1 + 2, t.getFloat(idx2 + rowsl));
                            a.setFloat(idx1 + 3, t.getFloat(idx2 + 2 * rowsl));
                        }
                    }
                } else if (columnsl == 2) {
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = idx0 + r * rowStridel;
                        t.setFloat(r, a.getFloat(idx1));
                        t.setFloat(rowsl + r, a.getFloat(idx1 + 1));
                    }
                    dctRows.forward(t, 0, scale);
                    dctRows.forward(t, rowsl, scale);
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = idx0 + r * rowStridel;
                        a.setFloat(idx1, t.getFloat(r));
                        a.setFloat(idx1 + 1, t.getFloat(rowsl + r));
                    }
                }
            }
        } else {
            for (long s = 0; s < slicesl; s++) {
                idx0 = s * sliceStridel;
                for (long r = 0; r < rowsl; r++) {
                    dctColumns.inverse(a, idx0 + r * rowStridel, scale);
                }
                if (columnsl > 2) {
                    for (long c = 0; c < columnsl; c += 4) {
                        for (long r = 0; r < rowsl; r++) {
                            idx1 = idx0 + r * rowStridel + c;
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
                            idx1 = idx0 + r * rowStridel + c;
                            idx2 = rowsl + r;
                            a.setFloat(idx1, t.getFloat(r));
                            a.setFloat(idx1 + 1, t.getFloat(idx2));
                            a.setFloat(idx1 + 2, t.getFloat(idx2 + rowsl));
                            a.setFloat(idx1 + 3, t.getFloat(idx2 + 2 * rowsl));
                        }
                    }
                } else if (columnsl == 2) {
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = idx0 + r * rowStridel;
                        t.setFloat(r, a.getFloat(idx1));
                        t.setFloat(rowsl + r, a.getFloat(idx1 + 1));
                    }
                    dctRows.inverse(t, 0, scale);
                    dctRows.inverse(t, rowsl, scale);
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = idx0 + r * rowStridel;
                        a.setFloat(idx1, t.getFloat(r));
                        a.setFloat(idx1 + 1, t.getFloat(rowsl + r));
                    }
                }
            }
        }
    }

    private void ddxt3da_sub(int isgn, float[][][] a, boolean scale)
    {
        int idx2;
        int nt = 4 * rows;
        if (columns == 2) {
            nt >>= 1;
        }
        float[] t = new float[nt];
        if (isgn == -1) {
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    dctColumns.forward(a[s][r], scale);
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
                        dctRows.forward(t, 0, scale);
                        dctRows.forward(t, rows, scale);
                        dctRows.forward(t, 2 * rows, scale);
                        dctRows.forward(t, 3 * rows, scale);
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
                    dctRows.forward(t, 0, scale);
                    dctRows.forward(t, rows, scale);
                    for (int r = 0; r < rows; r++) {
                        a[s][r][0] = t[r];
                        a[s][r][1] = t[rows + r];
                    }
                }
            }
        } else {
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    dctColumns.inverse(a[s][r], scale);
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
                        dctRows.inverse(t, 0, scale);
                        dctRows.inverse(t, rows, scale);
                        dctRows.inverse(t, 2 * rows, scale);
                        dctRows.inverse(t, 3 * rows, scale);
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
                    dctRows.inverse(t, 0, scale);
                    dctRows.inverse(t, rows, scale);
                    for (int r = 0; r < rows; r++) {
                        a[s][r][0] = t[r];
                        a[s][r][1] = t[rows + r];
                    }
                }
            }
        }
    }

    private void ddxt3db_sub(int isgn, float[] a, boolean scale)
    {
        int idx0, idx1, idx2;
        int nt = 4 * slices;
        if (columns == 2) {
            nt >>= 1;
        }
        float[] t = new float[nt];
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
                        dctSlices.forward(t, 0, scale);
                        dctSlices.forward(t, slices, scale);
                        dctSlices.forward(t, 2 * slices, scale);
                        dctSlices.forward(t, 3 * slices, scale);
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
                    dctSlices.forward(t, 0, scale);
                    dctSlices.forward(t, slices, scale);
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
                    dctSlices.inverse(t, 0, scale);
                    dctSlices.inverse(t, slices, scale);
                    dctSlices.inverse(t, 2 * slices, scale);
                    dctSlices.inverse(t, 3 * slices, scale);

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
                dctSlices.inverse(t, 0, scale);
                dctSlices.inverse(t, slices, scale);
                for (int s = 0; s < slices; s++) {
                    idx1 = s * sliceStride + idx0;
                    a[idx1] = t[s];
                    a[idx1 + 1] = t[slices + s];
                }
            }
        }
    }

    private void ddxt3db_sub(int isgn, FloatLargeArray a, boolean scale)
    {
        long idx0, idx1, idx2;
        long nt = 4 * slicesl;
        if (columnsl == 2) {
            nt >>= 1;
        }
        FloatLargeArray t = new FloatLargeArray(nt);
        if (isgn == -1) {
            if (columnsl > 2) {
                for (long r = 0; r < rowsl; r++) {
                    idx0 = r * rowStridel;
                    for (long c = 0; c < columnsl; c += 4) {
                        for (long s = 0; s < slicesl; s++) {
                            idx1 = s * sliceStridel + idx0 + c;
                            idx2 = slicesl + s;
                            t.setFloat(s, a.getFloat(idx1));
                            t.setFloat(idx2, a.getFloat(idx1 + 1));
                            t.setFloat(idx2 + slicesl, a.getFloat(idx1 + 2));
                            t.setFloat(idx2 + 2 * slicesl, a.getFloat(idx1 + 3));
                        }
                        dctSlices.forward(t, 0, scale);
                        dctSlices.forward(t, slicesl, scale);
                        dctSlices.forward(t, 2 * slicesl, scale);
                        dctSlices.forward(t, 3 * slicesl, scale);
                        for (long s = 0; s < slicesl; s++) {
                            idx1 = s * sliceStridel + idx0 + c;
                            idx2 = slicesl + s;
                            a.setFloat(idx1, t.getFloat(s));
                            a.setFloat(idx1 + 1, t.getFloat(idx2));
                            a.setFloat(idx1 + 2, t.getFloat(idx2 + slicesl));
                            a.setFloat(idx1 + 3, t.getFloat(idx2 + 2 * slicesl));
                        }
                    }
                }
            } else if (columnsl == 2) {
                for (long r = 0; r < rowsl; r++) {
                    idx0 = r * rowStridel;
                    for (long s = 0; s < slicesl; s++) {
                        idx1 = s * sliceStridel + idx0;
                        t.setFloat(s, a.getFloat(idx1));
                        t.setFloat(slicesl + s, a.getFloat(idx1 + 1));
                    }
                    dctSlices.forward(t, 0, scale);
                    dctSlices.forward(t, slicesl, scale);
                    for (long s = 0; s < slicesl; s++) {
                        idx1 = s * sliceStridel + idx0;
                        a.setFloat(idx1, t.getFloat(s));
                        a.setFloat(idx1 + 1, t.getFloat(slicesl + s));
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
                        t.setFloat(s, a.getFloat(idx1));
                        t.setFloat(idx2, a.getFloat(idx1 + 1));
                        t.setFloat(idx2 + slicesl, a.getFloat(idx1 + 2));
                        t.setFloat(idx2 + 2 * slicesl, a.getFloat(idx1 + 3));
                    }
                    dctSlices.inverse(t, 0, scale);
                    dctSlices.inverse(t, slicesl, scale);
                    dctSlices.inverse(t, 2 * slicesl, scale);
                    dctSlices.inverse(t, 3 * slicesl, scale);

                    for (long s = 0; s < slicesl; s++) {
                        idx1 = s * sliceStridel + idx0 + c;
                        idx2 = slicesl + s;
                        a.setFloat(idx1, t.getFloat(s));
                        a.setFloat(idx1 + 1, t.getFloat(idx2));
                        a.setFloat(idx1 + 2, t.getFloat(idx2 + slicesl));
                        a.setFloat(idx1 + 3, t.getFloat(idx2 + 2 * slicesl));
                    }
                }
            }
        } else if (columnsl == 2) {
            for (long r = 0; r < rowsl; r++) {
                idx0 = r * rowStridel;
                for (long s = 0; s < slicesl; s++) {
                    idx1 = s * sliceStridel + idx0;
                    t.setFloat(s, a.getFloat(idx1));
                    t.setFloat(slicesl + s, a.getFloat(idx1 + 1));
                }
                dctSlices.inverse(t, 0, scale);
                dctSlices.inverse(t, slicesl, scale);
                for (long s = 0; s < slicesl; s++) {
                    idx1 = s * sliceStridel + idx0;
                    a.setFloat(idx1, t.getFloat(s));
                    a.setFloat(idx1 + 1, t.getFloat(slicesl + s));
                }
            }
        }
    }

    private void ddxt3db_sub(int isgn, float[][][] a, boolean scale)
    {
        int idx2;
        int nt = 4 * slices;
        if (columns == 2) {
            nt >>= 1;
        }
        float[] t = new float[nt];
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
                        dctSlices.forward(t, 0, scale);
                        dctSlices.forward(t, slices, scale);
                        dctSlices.forward(t, 2 * slices, scale);
                        dctSlices.forward(t, 3 * slices, scale);
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
                    dctSlices.forward(t, 0, scale);
                    dctSlices.forward(t, slices, scale);
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
                    dctSlices.inverse(t, 0, scale);
                    dctSlices.inverse(t, slices, scale);
                    dctSlices.inverse(t, 2 * slices, scale);
                    dctSlices.inverse(t, 3 * slices, scale);

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
                dctSlices.inverse(t, 0, scale);
                dctSlices.inverse(t, slices, scale);
                for (int s = 0; s < slices; s++) {
                    a[s][r][0] = t[s];
                    a[s][r][1] = t[slices + s];
                }
            }
        }
    }

    private void ddxt3da_subth(final int isgn, final float[] a, final boolean scale)
    {
        final int nthreads = ConcurrencyUtils.getNumberOfThreads() > slices ? slices : ConcurrencyUtils.getNumberOfThreads();
        int nt = 4 * rows;
        if (columns == 2) {
            nt >>= 1;
        }
        Future<?>[] futures = new Future[nthreads];
        final int ntf = nt;
        for (int i = 0; i < nthreads; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {

                public void run()
                {
                    int idx0, idx1, idx2;
                    float[] t = new float[ntf];
                    if (isgn == -1) {
                        for (int s = n0; s < slices; s += nthreads) {
                            idx0 = s * sliceStride;
                            for (int r = 0; r < rows; r++) {
                                dctColumns.forward(a, idx0 + r * rowStride, scale);
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
                                    dctRows.forward(t, 0, scale);
                                    dctRows.forward(t, rows, scale);
                                    dctRows.forward(t, 2 * rows, scale);
                                    dctRows.forward(t, 3 * rows, scale);
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
                                dctRows.forward(t, 0, scale);
                                dctRows.forward(t, rows, scale);
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
                                dctColumns.inverse(a, idx0 + r * rowStride, scale);
                            }
                            if (columns > 2) {
                                for (int c = 0; c < columns; c += 4) {
                                    for (int j = 0; j < rows; j++) {
                                        idx1 = idx0 + j * rowStride + c;
                                        idx2 = rows + j;
                                        t[j] = a[idx1];
                                        t[idx2] = a[idx1 + 1];
                                        t[idx2 + rows] = a[idx1 + 2];
                                        t[idx2 + 2 * rows] = a[idx1 + 3];
                                    }
                                    dctRows.inverse(t, 0, scale);
                                    dctRows.inverse(t, rows, scale);
                                    dctRows.inverse(t, 2 * rows, scale);
                                    dctRows.inverse(t, 3 * rows, scale);
                                    for (int j = 0; j < rows; j++) {
                                        idx1 = idx0 + j * rowStride + c;
                                        idx2 = rows + j;
                                        a[idx1] = t[j];
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
                                dctRows.inverse(t, 0, scale);
                                dctRows.inverse(t, rows, scale);
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
            Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void ddxt3da_subth(final int isgn, final FloatLargeArray a, final boolean scale)
    {
        final int nthreads = (int) (ConcurrencyUtils.getNumberOfThreads() > slicesl ? slicesl : ConcurrencyUtils.getNumberOfThreads());
        long nt = 4 * rowsl;
        if (columnsl == 2) {
            nt >>= 1;
        }
        Future<?>[] futures = new Future[nthreads];
        final long ntf = nt;
        for (int i = 0; i < nthreads; i++) {
            final long n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {

                public void run()
                {
                    long idx0, idx1, idx2;
                    FloatLargeArray t = new FloatLargeArray(ntf);
                    if (isgn == -1) {
                        for (long s = n0; s < slicesl; s += nthreads) {
                            idx0 = s * sliceStridel;
                            for (long r = 0; r < rowsl; r++) {
                                dctColumns.forward(a, idx0 + r * rowStridel, scale);
                            }
                            if (columnsl > 2) {
                                for (long c = 0; c < columnsl; c += 4) {
                                    for (long r = 0; r < rowsl; r++) {
                                        idx1 = idx0 + r * rowStridel + c;
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
                                        idx1 = idx0 + r * rowStridel + c;
                                        idx2 = rowsl + r;
                                        a.setFloat(idx1, t.getFloat(r));
                                        a.setFloat(idx1 + 1, t.getFloat(idx2));
                                        a.setFloat(idx1 + 2, t.getFloat(idx2 + rowsl));
                                        a.setFloat(idx1 + 3, t.getFloat(idx2 + 2 * rowsl));
                                    }
                                }
                            } else if (columnsl == 2) {
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStridel;
                                    t.setFloat(r, a.getFloat(idx1));
                                    t.setFloat(rowsl + r, a.getFloat(idx1 + 1));
                                }
                                dctRows.forward(t, 0, scale);
                                dctRows.forward(t, rowsl, scale);
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStridel;
                                    a.setFloat(idx1, t.getFloat(r));
                                    a.setFloat(idx1 + 1, t.getFloat(rowsl + r));
                                }
                            }
                        }
                    } else {
                        for (long s = n0; s < slicesl; s += nthreads) {
                            idx0 = s * sliceStridel;
                            for (long r = 0; r < rowsl; r++) {
                                dctColumns.inverse(a, idx0 + r * rowStridel, scale);
                            }
                            if (columnsl > 2) {
                                for (long c = 0; c < columnsl; c += 4) {
                                    for (long j = 0; j < rowsl; j++) {
                                        idx1 = idx0 + j * rowStridel + c;
                                        idx2 = rowsl + j;
                                        t.setFloat(j, a.getFloat(idx1));
                                        t.setFloat(idx2, a.getFloat(idx1 + 1));
                                        t.setFloat(idx2 + rowsl, a.getFloat(idx1 + 2));
                                        t.setFloat(idx2 + 2 * rowsl, a.getFloat(idx1 + 3));
                                    }
                                    dctRows.inverse(t, 0, scale);
                                    dctRows.inverse(t, rowsl, scale);
                                    dctRows.inverse(t, 2 * rowsl, scale);
                                    dctRows.inverse(t, 3 * rowsl, scale);
                                    for (long j = 0; j < rowsl; j++) {
                                        idx1 = idx0 + j * rowStridel + c;
                                        idx2 = rowsl + j;
                                        a.setFloat(idx1, t.getFloat(j));
                                        a.setFloat(idx1 + 1, t.getFloat(idx2));
                                        a.setFloat(idx1 + 2, t.getFloat(idx2 + rowsl));
                                        a.setFloat(idx1 + 3, t.getFloat(idx2 + 2 * rowsl));
                                    }
                                }
                            } else if (columnsl == 2) {
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStridel;
                                    t.setFloat(r, a.getFloat(idx1));
                                    t.setFloat(rowsl + r, a.getFloat(idx1 + 1));
                                }
                                dctRows.inverse(t, 0, scale);
                                dctRows.inverse(t, rowsl, scale);
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStridel;
                                    a.setFloat(idx1, t.getFloat(r));
                                    a.setFloat(idx1 + 1, t.getFloat(rowsl + r));
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
            Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void ddxt3da_subth(final int isgn, final float[][][] a, final boolean scale)
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
                    float[] t = new float[ntf];
                    if (isgn == -1) {
                        for (int s = n0; s < slices; s += nthreads) {
                            for (int r = 0; r < rows; r++) {
                                dctColumns.forward(a[s][r], scale);
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
                                    dctRows.forward(t, 0, scale);
                                    dctRows.forward(t, rows, scale);
                                    dctRows.forward(t, 2 * rows, scale);
                                    dctRows.forward(t, 3 * rows, scale);
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
                                dctRows.forward(t, 0, scale);
                                dctRows.forward(t, rows, scale);
                                for (int r = 0; r < rows; r++) {
                                    a[s][r][0] = t[r];
                                    a[s][r][1] = t[rows + r];
                                }
                            }
                        }
                    } else {
                        for (int s = n0; s < slices; s += nthreads) {
                            for (int r = 0; r < rows; r++) {
                                dctColumns.inverse(a[s][r], scale);
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
                                    dctRows.inverse(t, 0, scale);
                                    dctRows.inverse(t, rows, scale);
                                    dctRows.inverse(t, 2 * rows, scale);
                                    dctRows.inverse(t, 3 * rows, scale);
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
                                dctRows.inverse(t, 0, scale);
                                dctRows.inverse(t, rows, scale);
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
            Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void ddxt3db_subth(final int isgn, final float[] a, final boolean scale)
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
                    float[] t = new float[ntf];
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
                                    dctSlices.forward(t, 0, scale);
                                    dctSlices.forward(t, slices, scale);
                                    dctSlices.forward(t, 2 * slices, scale);
                                    dctSlices.forward(t, 3 * slices, scale);
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
                                dctSlices.forward(t, 0, scale);
                                dctSlices.forward(t, slices, scale);
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
                                dctSlices.inverse(t, 0, scale);
                                dctSlices.inverse(t, slices, scale);
                                dctSlices.inverse(t, 2 * slices, scale);
                                dctSlices.inverse(t, 3 * slices, scale);
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
                            dctSlices.inverse(t, 0, scale);
                            dctSlices.inverse(t, slices, scale);

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
            Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void ddxt3db_subth(final int isgn, final FloatLargeArray a, final boolean scale)
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
                    FloatLargeArray t = new FloatLargeArray(ntf);
                    if (isgn == -1) {
                        if (columnsl > 2) {
                            for (long r = n0; r < rowsl; r += nthreads) {
                                idx0 = r * rowStridel;
                                for (long c = 0; c < columnsl; c += 4) {
                                    for (long s = 0; s < slicesl; s++) {
                                        idx1 = s * sliceStridel + idx0 + c;
                                        idx2 = slicesl + s;
                                        t.setFloat(s, a.getFloat(idx1));
                                        t.setFloat(idx2, a.getFloat(idx1 + 1));
                                        t.setFloat(idx2 + slicesl, a.getFloat(idx1 + 2));
                                        t.setFloat(idx2 + 2 * slicesl, a.getFloat(idx1 + 3));
                                    }
                                    dctSlices.forward(t, 0, scale);
                                    dctSlices.forward(t, slicesl, scale);
                                    dctSlices.forward(t, 2 * slicesl, scale);
                                    dctSlices.forward(t, 3 * slicesl, scale);
                                    for (long s = 0; s < slicesl; s++) {
                                        idx1 = s * sliceStridel + idx0 + c;
                                        idx2 = slicesl + s;
                                        a.setFloat(idx1, t.getFloat(s));
                                        a.setFloat(idx1 + 1, t.getFloat(idx2));
                                        a.setFloat(idx1 + 2, t.getFloat(idx2 + slicesl));
                                        a.setFloat(idx1 + 3, t.getFloat(idx2 + 2 * slicesl));
                                    }
                                }
                            }
                        } else if (columnsl == 2) {
                            for (long r = n0; r < rowsl; r += nthreads) {
                                idx0 = r * rowStridel;
                                for (long s = 0; s < slicesl; s++) {
                                    idx1 = s * sliceStridel + idx0;
                                    t.setFloat(s, a.getFloat(idx1));
                                    t.setFloat(slicesl + s, a.getFloat(idx1 + 1));
                                }
                                dctSlices.forward(t, 0, scale);
                                dctSlices.forward(t, slicesl, scale);
                                for (long s = 0; s < slicesl; s++) {
                                    idx1 = s * sliceStridel + idx0;
                                    a.setFloat(idx1, t.getFloat(s));
                                    a.setFloat(idx1 + 1, t.getFloat(slicesl + s));
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
                                    t.setFloat(s, a.getFloat(idx1));
                                    t.setFloat(idx2, a.getFloat(idx1 + 1));
                                    t.setFloat(idx2 + slicesl, a.getFloat(idx1 + 2));
                                    t.setFloat(idx2 + 2 * slicesl, a.getFloat(idx1 + 3));
                                }
                                dctSlices.inverse(t, 0, scale);
                                dctSlices.inverse(t, slicesl, scale);
                                dctSlices.inverse(t, 2 * slicesl, scale);
                                dctSlices.inverse(t, 3 * slicesl, scale);
                                for (long s = 0; s < slicesl; s++) {
                                    idx1 = s * sliceStridel + idx0 + c;
                                    idx2 = slicesl + s;
                                    a.setFloat(idx1, t.getFloat(s));
                                    a.setFloat(idx1 + 1, t.getFloat(idx2));
                                    a.setFloat(idx1 + 2, t.getFloat(idx2 + slicesl));
                                    a.setFloat(idx1 + 3, t.getFloat(idx2 + 2 * slicesl));
                                }
                            }
                        }
                    } else if (columnsl == 2) {
                        for (long r = n0; r < rowsl; r += nthreads) {
                            idx0 = r * rowStridel;
                            for (long s = 0; s < slicesl; s++) {
                                idx1 = s * sliceStridel + idx0;
                                t.setFloat(s, a.getFloat(idx1));
                                t.setFloat(slicesl + s, a.getFloat(idx1 + 1));
                            }
                            dctSlices.inverse(t, 0, scale);
                            dctSlices.inverse(t, slicesl, scale);

                            for (long s = 0; s < slicesl; s++) {
                                idx1 = s * sliceStridel + idx0;
                                a.setFloat(idx1, t.getFloat(s));
                                a.setFloat(idx1 + 1, t.getFloat(slicesl + s));
                            }
                        }
                    }
                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void ddxt3db_subth(final int isgn, final float[][][] a, final boolean scale)
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
                    float[] t = new float[ntf];
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
                                    dctSlices.forward(t, 0, scale);
                                    dctSlices.forward(t, slices, scale);
                                    dctSlices.forward(t, 2 * slices, scale);
                                    dctSlices.forward(t, 3 * slices, scale);
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
                                dctSlices.forward(t, 0, scale);
                                dctSlices.forward(t, slices, scale);
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
                                dctSlices.inverse(t, 0, scale);
                                dctSlices.inverse(t, slices, scale);
                                dctSlices.inverse(t, 2 * slices, scale);
                                dctSlices.inverse(t, 3 * slices, scale);
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
                            dctSlices.inverse(t, 0, scale);
                            dctSlices.inverse(t, slices, scale);

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
            Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(FloatDCT_3D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
