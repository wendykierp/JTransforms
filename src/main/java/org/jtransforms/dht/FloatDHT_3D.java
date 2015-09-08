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
import pl.edu.icm.jlargearrays.FloatLargeArray;
import pl.edu.icm.jlargearrays.LargeArray;

/**
 * Computes 3D Discrete Hartley Transform (DHT) of real, single precision data.
 * The sizes of all three dimensions can be arbitrary numbers. This is a
 * parallel implementation optimized for SMP systems.<br>
 * <br>
 * Part of code is derived from General Purpose FFT Package written by Takuya
 * Ooura (http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html)
 *  
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class FloatDHT_3D
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

    private FloatDHT_1D dhtSlices, dhtRows, dhtColumns;

    private boolean isPowerOfTwo = false;

    private boolean useThreads = false;

    /**
     * Creates new instance of FloatDHT_3D.
     *  
     * @param slices  number of slices
     * @param rows    number of rows
     * @param columns number of columns
     */
    public FloatDHT_3D(long slices, long rows, long columns)
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
        dhtSlices = new FloatDHT_1D(slices);
        if (slices == rows) {
            dhtRows = dhtSlices;
        } else {
            dhtRows = new FloatDHT_1D(rows);
        }
        if (slices == columns) {
            dhtColumns = dhtSlices;
        } else if (rows == columns) {
            dhtColumns = dhtRows;
        } else {
            dhtColumns = new FloatDHT_1D(columns);
        }
    }

    /**
     * Computes the 3D real, forward DHT leaving the result in <code>a</code>.
     * The data is stored in 1D array addressed in slice-major, then row-major,
     * then column-major, in order of significance, i.e. the element (i,j,k) of
     * 3D array x[slices][rows][columns] is stored in a[i*sliceStride +
     * j*rowStride + k], where sliceStride = rows * columns and rowStride =
     * columns.
     *  
     * @param a data to transform
     */
    public void forward(final float[] a)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            if ((nthreads > 1) && useThreads) {
                ddxt3da_subth(-1, a, true);
                ddxt3db_subth(-1, a, true);
            } else {
                ddxt3da_sub(-1, a, true);
                ddxt3db_sub(-1, a, true);
            }
            yTransform(a);
        } else {
            if ((nthreads > 1) && useThreads && (slices >= nthreads) && (rows >= nthreads) && (columns >= nthreads)) {
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
                                    dhtColumns.forward(a, idx1 + r * rowStride);
                                }
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
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
                                    dhtRows.forward(temp);
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
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
                }

                p = rows / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final int startRow = l * p;
                    final int stopRow;
                    if (l == nthreads - 1) {
                        stopRow = rows;
                    } else {
                        stopRow = startRow + p;
                    }
                    futures[l] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            float[] temp = new float[slices];
                            for (int r = startRow; r < stopRow; r++) {
                                int idx1 = r * rowStride;
                                for (int c = 0; c < columns; c++) {
                                    for (int s = 0; s < slices; s++) {
                                        int idx3 = s * sliceStride + idx1 + c;
                                        temp[s] = a[idx3];
                                    }
                                    dhtSlices.forward(temp);
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
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
                }

            } else {
                for (int s = 0; s < slices; s++) {
                    int idx1 = s * sliceStride;
                    for (int r = 0; r < rows; r++) {
                        dhtColumns.forward(a, idx1 + r * rowStride);
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
                        dhtRows.forward(temp);
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
                        dhtSlices.forward(temp);
                        for (int s = 0; s < slices; s++) {
                            int idx3 = s * sliceStride + idx1 + c;
                            a[idx3] = temp[s];
                        }
                    }
                }
            }
            yTransform(a);
        }
    }

    /**
     * Computes the 3D real, forward DHT leaving the result in <code>a</code>.
     * The data is stored in 1D array addressed in slice-major, then row-major,
     * then column-major, in order of significance, i.e. the element (i,j,k) of
     * 3D array x[slices][rows][columns] is stored in a[i*sliceStride +
     * j*rowStride + k], where sliceStride = rows * columns and rowStride =
     * columns.
     *  
     * @param a data to transform
     */
    public void forward(final FloatLargeArray a)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            if ((nthreads > 1) && useThreads) {
                ddxt3da_subth(-1, a, true);
                ddxt3db_subth(-1, a, true);
            } else {
                ddxt3da_sub(-1, a, true);
                ddxt3db_sub(-1, a, true);
            }
            yTransform(a);
        } else {
            if ((nthreads > 1) && useThreads && (slicesl >= nthreads) && (rowsl >= nthreads) && (columnsl >= nthreads)) {
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
                                    dhtColumns.forward(a, idx1 + r * rowStride);
                                }
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
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
                                long idx1 = s * sliceStride;
                                for (long c = 0; c < columnsl; c++) {
                                    for (long r = 0; r < rowsl; r++) {
                                        long idx3 = idx1 + r * rowStride + c;
                                        temp.setFloat(r, a.getFloat(idx3));
                                    }
                                    dhtRows.forward(temp);
                                    for (long r = 0; r < rowsl; r++) {
                                        long idx3 = idx1 + r * rowStride + c;
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
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
                }

                p = rowsl / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final long startRow = l * p;
                    final long stopRow;
                    if (l == nthreads - 1) {
                        stopRow = rowsl;
                    } else {
                        stopRow = startRow + p;
                    }
                    futures[l] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            FloatLargeArray temp = new FloatLargeArray(slicesl, false);
                            for (long r = startRow; r < stopRow; r++) {
                                long idx1 = r * rowStride;
                                for (long c = 0; c < columnsl; c++) {
                                    for (long s = 0; s < slicesl; s++) {
                                        long idx3 = s * sliceStride + idx1 + c;
                                        temp.setFloat(s, a.getFloat(idx3));
                                    }
                                    dhtSlices.forward(temp);
                                    for (long s = 0; s < slicesl; s++) {
                                        long idx3 = s * sliceStride + idx1 + c;
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
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
                }

            } else {
                for (long s = 0; s < slicesl; s++) {
                    long idx1 = s * sliceStride;
                    for (long r = 0; r < rowsl; r++) {
                        dhtColumns.forward(a, idx1 + r * rowStride);
                    }
                }
                FloatLargeArray temp = new FloatLargeArray(rowsl, false);
                for (long s = 0; s < slicesl; s++) {
                    long idx1 = s * sliceStride;
                    for (long c = 0; c < columnsl; c++) {
                        for (long r = 0; r < rowsl; r++) {
                            long idx3 = idx1 + r * rowStride + c;
                            temp.setFloat(r, a.getFloat(idx3));
                        }
                        dhtRows.forward(temp);
                        for (long r = 0; r < rowsl; r++) {
                            long idx3 = idx1 + r * rowStride + c;
                            a.setFloat(idx3, temp.getFloat(r));
                        }
                    }
                }
                temp = new FloatLargeArray(slicesl, false);
                for (long r = 0; r < rowsl; r++) {
                    long idx1 = r * rowStride;
                    for (long c = 0; c < columnsl; c++) {
                        for (long s = 0; s < slicesl; s++) {
                            long idx3 = s * sliceStride + idx1 + c;
                            temp.setFloat(s, a.getFloat(idx3));
                        }
                        dhtSlices.forward(temp);
                        for (long s = 0; s < slicesl; s++) {
                            long idx3 = s * sliceStride + idx1 + c;
                            a.setFloat(idx3, temp.getFloat(s));
                        }
                    }
                }
            }
            yTransform(a);
        }
    }

    /**
     * Computes the 3D real, forward DHT leaving the result in <code>a</code>.
     * The data is stored in 3D array.
     *  
     * @param a data to transform
     */
    public void forward(final float[][][] a)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            if ((nthreads > 1) && useThreads) {
                ddxt3da_subth(-1, a, true);
                ddxt3db_subth(-1, a, true);
            } else {
                ddxt3da_sub(-1, a, true);
                ddxt3db_sub(-1, a, true);
            }
            yTransform(a);
        } else {
            if ((nthreads > 1) && useThreads && (slices >= nthreads) && (rows >= nthreads) && (columns >= nthreads)) {
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
                                    dhtColumns.forward(a[s][r]);
                                }
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
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
                                    dhtRows.forward(temp);
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
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
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
                                    dhtSlices.forward(temp);
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
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
                }

            } else {
                for (int s = 0; s < slices; s++) {
                    for (int r = 0; r < rows; r++) {
                        dhtColumns.forward(a[s][r]);
                    }
                }
                float[] temp = new float[rows];
                for (int s = 0; s < slices; s++) {
                    for (int c = 0; c < columns; c++) {
                        for (int r = 0; r < rows; r++) {
                            temp[r] = a[s][r][c];
                        }
                        dhtRows.forward(temp);
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
                        dhtSlices.forward(temp);
                        for (int s = 0; s < slices; s++) {
                            a[s][r][c] = temp[s];
                        }
                    }
                }
            }
            yTransform(a);
        }
    }

    /**
     * Computes the 3D real, inverse DHT leaving the result in <code>a</code>.
     * The data is stored in 1D array addressed in slice-major, then row-major,
     * then column-major, in order of significance, i.e. the element (i,j,k) of
     * 3D array x[slices][rows][columns] is stored in a[i*sliceStride +
     * j*rowStride + k], where sliceStride = rows * columns and rowStride =
     * columns.
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
            yTransform(a);
        } else {
            if ((nthreads > 1) && useThreads && (slices >= nthreads) && (rows >= nthreads) && (columns >= nthreads)) {
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
                                    dhtColumns.inverse(a, idx1 + r * rowStride, scale);
                                }
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
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
                                    dhtRows.inverse(temp, scale);
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
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
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
                                    dhtSlices.inverse(temp, scale);
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
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
                }

            } else {
                for (int s = 0; s < slices; s++) {
                    int idx1 = s * sliceStride;
                    for (int r = 0; r < rows; r++) {
                        dhtColumns.inverse(a, idx1 + r * rowStride, scale);
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
                        dhtRows.inverse(temp, scale);
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
                        dhtSlices.inverse(temp, scale);
                        for (int s = 0; s < slices; s++) {
                            int idx3 = s * sliceStride + idx1 + c;
                            a[idx3] = temp[s];
                        }
                    }
                }
            }
            yTransform(a);
        }
    }

    /**
     * Computes the 3D real, inverse DHT leaving the result in <code>a</code>.
     * The data is stored in 1D array addressed in slice-major, then row-major,
     * then column-major, in order of significance, i.e. the element (i,j,k) of
     * 3D array x[slices][rows][columns] is stored in a[i*sliceStride +
     * j*rowStride + k], where sliceStride = rows * columns and rowStride =
     * columns.
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
            yTransform(a);
        } else {
            if ((nthreads > 1) && useThreads && (slicesl >= nthreads) && (rowsl >= nthreads) && (columnsl >= nthreads)) {
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
                                    dhtColumns.inverse(a, idx1 + r * rowStridel, scale);
                                }
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
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
                                    dhtRows.inverse(temp, scale);
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
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
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
                                    dhtSlices.inverse(temp, scale);
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
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
                }

            } else {
                for (long s = 0; s < slicesl; s++) {
                    long idx1 = s * sliceStridel;
                    for (long r = 0; r < rowsl; r++) {
                        dhtColumns.inverse(a, idx1 + r * rowStridel, scale);
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
                        dhtRows.inverse(temp, scale);
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
                        dhtSlices.inverse(temp, scale);
                        for (long s = 0; s < slicesl; s++) {
                            long idx3 = s * sliceStridel + idx1 + c;
                            a.setFloat(idx3, temp.getFloat(s));
                        }
                    }
                }
            }
            yTransform(a);
        }
    }

    /**
     * Computes the 3D real, inverse DHT leaving the result in <code>a</code>.
     * The data is stored in 3D array.
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
            yTransform(a);
        } else {
            if ((nthreads > 1) && useThreads && (slices >= nthreads) && (rows >= nthreads) && (columns >= nthreads)) {
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
                                    dhtColumns.inverse(a[s][r], scale);
                                }
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
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
                                    dhtRows.inverse(temp, scale);
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
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
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
                                    dhtSlices.inverse(temp, scale);
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
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
                }

            } else {
                for (int s = 0; s < slices; s++) {
                    for (int r = 0; r < rows; r++) {
                        dhtColumns.inverse(a[s][r], scale);
                    }
                }
                float[] temp = new float[rows];
                for (int s = 0; s < slices; s++) {
                    for (int c = 0; c < columns; c++) {
                        for (int r = 0; r < rows; r++) {
                            temp[r] = a[s][r][c];
                        }
                        dhtRows.inverse(temp, scale);
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
                        dhtSlices.inverse(temp, scale);
                        for (int s = 0; s < slices; s++) {
                            a[s][r][c] = temp[s];
                        }
                    }
                }
            }
            yTransform(a);
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
                    dhtColumns.forward(a, idx0 + r * rowStride);
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
                        dhtRows.forward(t, 0);
                        dhtRows.forward(t, rows);
                        dhtRows.forward(t, 2 * rows);
                        dhtRows.forward(t, 3 * rows);
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
                    dhtRows.forward(t, 0);
                    dhtRows.forward(t, rows);
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
                    dhtColumns.inverse(a, idx0 + r * rowStride, scale);
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
                        dhtRows.inverse(t, 0, scale);
                        dhtRows.inverse(t, rows, scale);
                        dhtRows.inverse(t, 2 * rows, scale);
                        dhtRows.inverse(t, 3 * rows, scale);
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
                    dhtRows.inverse(t, 0, scale);
                    dhtRows.inverse(t, rows, scale);
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
                    dhtColumns.forward(a, idx0 + r * rowStridel);
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
                        dhtRows.forward(t, 0);
                        dhtRows.forward(t, rowsl);
                        dhtRows.forward(t, 2 * rowsl);
                        dhtRows.forward(t, 3 * rowsl);
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
                    dhtRows.forward(t, 0);
                    dhtRows.forward(t, rowsl);
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
                    dhtColumns.inverse(a, idx0 + r * rowStridel, scale);
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
                        dhtRows.inverse(t, 0, scale);
                        dhtRows.inverse(t, rowsl, scale);
                        dhtRows.inverse(t, 2 * rowsl, scale);
                        dhtRows.inverse(t, 3 * rowsl, scale);
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
                    dhtRows.inverse(t, 0, scale);
                    dhtRows.inverse(t, rowsl, scale);
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
        if (columnsl == 2) {
            nt >>= 1;
        }
        float[] t = new float[nt];
        if (isgn == -1) {
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    dhtColumns.forward(a[s][r]);
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
                        dhtRows.forward(t, 0);
                        dhtRows.forward(t, rows);
                        dhtRows.forward(t, 2 * rows);
                        dhtRows.forward(t, 3 * rows);
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
                    dhtRows.forward(t, 0);
                    dhtRows.forward(t, rows);
                    for (int r = 0; r < rows; r++) {
                        a[s][r][0] = t[r];
                        a[s][r][1] = t[rows + r];
                    }
                }
            }
        } else {
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    dhtColumns.inverse(a[s][r], scale);
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
                        dhtRows.inverse(t, 0, scale);
                        dhtRows.inverse(t, rows, scale);
                        dhtRows.inverse(t, 2 * rows, scale);
                        dhtRows.inverse(t, 3 * rows, scale);
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
                    dhtRows.inverse(t, 0, scale);
                    dhtRows.inverse(t, rows, scale);
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
                        dhtSlices.forward(t, 0);
                        dhtSlices.forward(t, slices);
                        dhtSlices.forward(t, 2 * slices);
                        dhtSlices.forward(t, 3 * slices);
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
                    dhtSlices.forward(t, 0);
                    dhtSlices.forward(t, slices);
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
                    dhtSlices.inverse(t, 0, scale);
                    dhtSlices.inverse(t, slices, scale);
                    dhtSlices.inverse(t, 2 * slices, scale);
                    dhtSlices.inverse(t, 3 * slices, scale);

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
                dhtSlices.inverse(t, 0, scale);
                dhtSlices.inverse(t, slices, scale);
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
                        dhtSlices.forward(t, 0);
                        dhtSlices.forward(t, slicesl);
                        dhtSlices.forward(t, 2 * slicesl);
                        dhtSlices.forward(t, 3 * slicesl);
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
                    dhtSlices.forward(t, 0);
                    dhtSlices.forward(t, slicesl);
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
                    dhtSlices.inverse(t, 0, scale);
                    dhtSlices.inverse(t, slicesl, scale);
                    dhtSlices.inverse(t, 2 * slicesl, scale);
                    dhtSlices.inverse(t, 3 * slicesl, scale);

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
                dhtSlices.inverse(t, 0, scale);
                dhtSlices.inverse(t, slicesl, scale);
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
                        dhtSlices.forward(t, 0);
                        dhtSlices.forward(t, slices);
                        dhtSlices.forward(t, 2 * slices);
                        dhtSlices.forward(t, 3 * slices);
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
                    dhtSlices.forward(t, 0);
                    dhtSlices.forward(t, slices);
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
                    dhtSlices.inverse(t, 0, scale);
                    dhtSlices.inverse(t, slices, scale);
                    dhtSlices.inverse(t, 2 * slices, scale);
                    dhtSlices.inverse(t, 3 * slices, scale);

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
                dhtSlices.inverse(t, 0, scale);
                dhtSlices.inverse(t, slices, scale);
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
                        for (int s = n0; s < slices; s += nthreads) {
                            idx0 = s * sliceStride;
                            for (int r = 0; r < rows; r++) {
                                dhtColumns.forward(a, idx0 + r * rowStride);
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
                                    dhtRows.forward(t, 0);
                                    dhtRows.forward(t, rows);
                                    dhtRows.forward(t, 2 * rows);
                                    dhtRows.forward(t, 3 * rows);
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
                                dhtRows.forward(t, 0);
                                dhtRows.forward(t, rows);
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
                                dhtColumns.inverse(a, idx0 + r * rowStride, scale);
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
                                    dhtRows.inverse(t, 0, scale);
                                    dhtRows.inverse(t, rows, scale);
                                    dhtRows.inverse(t, 2 * rows, scale);
                                    dhtRows.inverse(t, 3 * rows, scale);
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
                                dhtRows.inverse(t, 0, scale);
                                dhtRows.inverse(t, rows, scale);
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
            Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void ddxt3da_subth(final int isgn, final FloatLargeArray a, final boolean scale)
    {
        final int nthreads = (int) (ConcurrencyUtils.getNumberOfThreads() > slicesl ? slicesl : ConcurrencyUtils.getNumberOfThreads());
        long nt = 4 * rowsl;
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
                        for (long s = n0; s < slicesl; s += nthreads) {
                            idx0 = s * sliceStride;
                            for (long r = 0; r < rowsl; r++) {
                                dhtColumns.forward(a, idx0 + r * rowStride);
                            }
                            if (columnsl > 2) {
                                for (long c = 0; c < columnsl; c += 4) {
                                    for (long r = 0; r < rowsl; r++) {
                                        idx1 = idx0 + r * rowStride + c;
                                        idx2 = rowsl + r;
                                        t.setFloat(r, a.getFloat(idx1));
                                        t.setFloat(idx2, a.getFloat(idx1 + 1));
                                        t.setFloat(idx2 + rowsl, a.getFloat(idx1 + 2));
                                        t.setFloat(idx2 + 2 * rowsl, a.getFloat(idx1 + 3));
                                    }
                                    dhtRows.forward(t, 0);
                                    dhtRows.forward(t, rowsl);
                                    dhtRows.forward(t, 2 * rowsl);
                                    dhtRows.forward(t, 3 * rowsl);
                                    for (long r = 0; r < rowsl; r++) {
                                        idx1 = idx0 + r * rowStride + c;
                                        idx2 = rowsl + r;
                                        a.setFloat(idx1, t.getFloat(r));
                                        a.setFloat(idx1 + 1, t.getFloat(idx2));
                                        a.setFloat(idx1 + 2, t.getFloat(idx2 + rowsl));
                                        a.setFloat(idx1 + 3, t.getFloat(idx2 + 2 * rowsl));
                                    }
                                }
                            } else if (columnsl == 2) {
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStride;
                                    t.setFloat(r, a.getFloat(idx1));
                                    t.setFloat(rowsl + r, a.getFloat(idx1 + 1));
                                }
                                dhtRows.forward(t, 0);
                                dhtRows.forward(t, rowsl);
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStride;
                                    a.setFloat(idx1, t.getFloat(r));
                                    a.setFloat(idx1 + 1, t.getFloat(rowsl + r));
                                }
                            }
                        }
                    } else {
                        for (long s = n0; s < slicesl; s += nthreads) {
                            idx0 = s * sliceStride;
                            for (long r = 0; r < rowsl; r++) {
                                dhtColumns.inverse(a, idx0 + r * rowStride, scale);
                            }
                            if (columnsl > 2) {
                                for (long c = 0; c < columnsl; c += 4) {
                                    for (long r = 0; r < rowsl; r++) {
                                        idx1 = idx0 + r * rowStride + c;
                                        idx2 = rowsl + r;
                                        t.setFloat(r, a.getFloat(idx1));
                                        t.setFloat(idx2, a.getFloat(idx1 + 1));
                                        t.setFloat(idx2 + rowsl, a.getFloat(idx1 + 2));
                                        t.setFloat(idx2 + 2 * rowsl, a.getFloat(idx1 + 3));
                                    }
                                    dhtRows.inverse(t, 0, scale);
                                    dhtRows.inverse(t, rowsl, scale);
                                    dhtRows.inverse(t, 2 * rowsl, scale);
                                    dhtRows.inverse(t, 3 * rowsl, scale);
                                    for (long r = 0; r < rowsl; r++) {
                                        idx1 = idx0 + r * rowStride + c;
                                        idx2 = rowsl + r;
                                        a.setFloat(idx1, t.getFloat(r));
                                        a.setFloat(idx1 + 1, t.getFloat(idx2));
                                        a.setFloat(idx1 + 2, t.getFloat(idx2 + rowsl));
                                        a.setFloat(idx1 + 3, t.getFloat(idx2 + 2 * rowsl));
                                    }
                                }
                            } else if (columnsl == 2) {
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStride;
                                    t.setFloat(r, a.getFloat(idx1));
                                    t.setFloat(rowsl + r, a.getFloat(idx1 + 1));
                                }
                                dhtRows.inverse(t, 0, scale);
                                dhtRows.inverse(t, rowsl, scale);
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStride;
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
            Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
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
                                dhtColumns.forward(a[s][r]);
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
                                    dhtRows.forward(t, 0);
                                    dhtRows.forward(t, rows);
                                    dhtRows.forward(t, 2 * rows);
                                    dhtRows.forward(t, 3 * rows);
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
                                dhtRows.forward(t, 0);
                                dhtRows.forward(t, rows);
                                for (int r = 0; r < rows; r++) {
                                    a[s][r][0] = t[r];
                                    a[s][r][1] = t[rows + r];
                                }
                            }
                        }
                    } else {
                        for (int s = n0; s < slices; s += nthreads) {
                            for (int r = 0; r < rows; r++) {
                                dhtColumns.inverse(a[s][r], scale);
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
                                    dhtRows.inverse(t, 0, scale);
                                    dhtRows.inverse(t, rows, scale);
                                    dhtRows.inverse(t, 2 * rows, scale);
                                    dhtRows.inverse(t, 3 * rows, scale);
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
                                dhtRows.inverse(t, 0, scale);
                                dhtRows.inverse(t, rows, scale);
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
            Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
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
                                    dhtSlices.forward(t, 0);
                                    dhtSlices.forward(t, slices);
                                    dhtSlices.forward(t, 2 * slices);
                                    dhtSlices.forward(t, 3 * slices);
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
                                dhtSlices.forward(t, 0);
                                dhtSlices.forward(t, slices);
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
                                dhtSlices.inverse(t, 0, scale);
                                dhtSlices.inverse(t, slices, scale);
                                dhtSlices.inverse(t, 2 * slices, scale);
                                dhtSlices.inverse(t, 3 * slices, scale);
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
                            dhtSlices.inverse(t, 0, scale);
                            dhtSlices.inverse(t, slices, scale);
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
            Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
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
                                    dhtSlices.forward(t, 0);
                                    dhtSlices.forward(t, slicesl);
                                    dhtSlices.forward(t, 2 * slicesl);
                                    dhtSlices.forward(t, 3 * slicesl);
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
                                dhtSlices.forward(t, 0);
                                dhtSlices.forward(t, slicesl);
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
                                dhtSlices.inverse(t, 0, scale);
                                dhtSlices.inverse(t, slicesl, scale);
                                dhtSlices.inverse(t, 2 * slicesl, scale);
                                dhtSlices.inverse(t, 3 * slicesl, scale);
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
                            dhtSlices.inverse(t, 0, scale);
                            dhtSlices.inverse(t, slicesl, scale);
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
            Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
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
                                    dhtSlices.forward(t, 0);
                                    dhtSlices.forward(t, slices);
                                    dhtSlices.forward(t, 2 * slices);
                                    dhtSlices.forward(t, 3 * slices);
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
                                dhtSlices.forward(t, 0);
                                dhtSlices.forward(t, slices);
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
                                dhtSlices.inverse(t, 0, scale);
                                dhtSlices.inverse(t, slices, scale);
                                dhtSlices.inverse(t, 2 * slices, scale);
                                dhtSlices.inverse(t, 3 * slices, scale);
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
                            dhtSlices.inverse(t, 0, scale);
                            dhtSlices.inverse(t, slices, scale);

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
            Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(FloatDHT_3D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void yTransform(float[] a)
    {
        float A, B, C, D, E, F, G, H;
        int cC, rC, sC;
        int idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, idx9, idx10, idx11, idx12;
        for (int s = 0; s <= slices / 2; s++) {
            sC = (slices - s) % slices;
            idx9 = s * sliceStride;
            idx10 = sC * sliceStride;
            for (int r = 0; r <= rows / 2; r++) {
                rC = (rows - r) % rows;
                idx11 = r * rowStride;
                idx12 = rC * rowStride;
                for (int c = 0; c <= columns / 2; c++) {
                    cC = (columns - c) % columns;
                    idx1 = idx9 + idx12 + c;
                    idx2 = idx9 + idx11 + cC;
                    idx3 = idx10 + idx11 + c;
                    idx4 = idx10 + idx12 + cC;
                    idx5 = idx10 + idx12 + c;
                    idx6 = idx10 + idx11 + cC;
                    idx7 = idx9 + idx11 + c;
                    idx8 = idx9 + idx12 + cC;
                    A = a[idx1];
                    B = a[idx2];
                    C = a[idx3];
                    D = a[idx4];
                    E = a[idx5];
                    F = a[idx6];
                    G = a[idx7];
                    H = a[idx8];
                    a[idx7] = (A + B + C - D) / 2;
                    a[idx3] = (E + F + G - H) / 2;
                    a[idx1] = (G + H + E - F) / 2;
                    a[idx5] = (C + D + A - B) / 2;
                    a[idx2] = (H + G + F - E) / 2;
                    a[idx6] = (D + C + B - A) / 2;
                    a[idx8] = (B + A + D - C) / 2;
                    a[idx4] = (F + E + H - G) / 2;
                }
            }
        }
    }

    private void yTransform(FloatLargeArray a)
    {
        float A, B, C, D, E, F, G, H;
        long cC, rC, sC;
        long idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, idx9, idx10, idx11, idx12;
        for (long s = 0; s <= slicesl / 2; s++) {
            sC = (slicesl - s) % slicesl;
            idx9 = s * sliceStridel;
            idx10 = sC * sliceStridel;
            for (long r = 0; r <= rowsl / 2; r++) {
                rC = (rowsl - r) % rowsl;
                idx11 = r * rowStridel;
                idx12 = rC * rowStridel;
                for (long c = 0; c <= columnsl / 2; c++) {
                    cC = (columnsl - c) % columnsl;
                    idx1 = idx9 + idx12 + c;
                    idx2 = idx9 + idx11 + cC;
                    idx3 = idx10 + idx11 + c;
                    idx4 = idx10 + idx12 + cC;
                    idx5 = idx10 + idx12 + c;
                    idx6 = idx10 + idx11 + cC;
                    idx7 = idx9 + idx11 + c;
                    idx8 = idx9 + idx12 + cC;
                    A = a.getFloat(idx1);
                    B = a.getFloat(idx2);
                    C = a.getFloat(idx3);
                    D = a.getFloat(idx4);
                    E = a.getFloat(idx5);
                    F = a.getFloat(idx6);
                    G = a.getFloat(idx7);
                    H = a.getFloat(idx8);
                    a.setFloat(idx7, (A + B + C - D) / 2);
                    a.setFloat(idx3, (E + F + G - H) / 2);
                    a.setFloat(idx1, (G + H + E - F) / 2);
                    a.setFloat(idx5, (C + D + A - B) / 2);
                    a.setFloat(idx2, (H + G + F - E) / 2);
                    a.setFloat(idx6, (D + C + B - A) / 2);
                    a.setFloat(idx8, (B + A + D - C) / 2);
                    a.setFloat(idx4, (F + E + H - G) / 2);
                }
            }
        }
    }

    private void yTransform(float[][][] a)
    {
        float A, B, C, D, E, F, G, H;
        int cC, rC, sC;
        for (int s = 0; s <= slices / 2; s++) {
            sC = (slices - s) % slices;
            for (int r = 0; r <= rows / 2; r++) {
                rC = (rows - r) % rows;
                for (int c = 0; c <= columns / 2; c++) {
                    cC = (columns - c) % columns;
                    A = a[s][rC][c];
                    B = a[s][r][cC];
                    C = a[sC][r][c];
                    D = a[sC][rC][cC];
                    E = a[sC][rC][c];
                    F = a[sC][r][cC];
                    G = a[s][r][c];
                    H = a[s][rC][cC];
                    a[s][r][c] = (A + B + C - D) / 2;
                    a[sC][r][c] = (E + F + G - H) / 2;
                    a[s][rC][c] = (G + H + E - F) / 2;
                    a[sC][rC][c] = (C + D + A - B) / 2;
                    a[s][r][cC] = (H + G + F - E) / 2;
                    a[sC][r][cC] = (D + C + B - A) / 2;
                    a[s][rC][cC] = (B + A + D - C) / 2;
                    a[sC][rC][cC] = (F + E + H - G) / 2;
                }
            }
        }
    }

}
