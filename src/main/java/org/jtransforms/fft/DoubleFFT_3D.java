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
package org.jtransforms.fft;

import java.util.concurrent.Future;
import java.util.concurrent.ExecutionException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.jtransforms.utils.CommonUtils;
import pl.edu.icm.jlargearrays.ConcurrencyUtils;
import pl.edu.icm.jlargearrays.DoubleLargeArray;
import pl.edu.icm.jlargearrays.LargeArray;
import pl.edu.icm.jlargearrays.LargeArrayUtils;
import static org.apache.commons.math3.util.FastMath.*;

/**
 * Computes 3D Discrete Fourier Transform (DFT) of complex and real, double
 * precision data. The sizes of all three dimensions can be arbitrary numbers.
 * This is a parallel implementation of split-radix and mixed-radix algorithms
 * optimized for SMP systems. <br>
 * <br>
 * Part of the code is derived from General Purpose FFT Package written by
 * Takuya Ooura (http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html)
 *  
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class DoubleFFT_3D
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

    private DoubleFFT_1D fftSlices, fftRows, fftColumns;

    private boolean isPowerOfTwo = false;

    private boolean useThreads = false;

    /**
     * Creates new instance of DoubleFFT_3D.
     *  
     * @param slices  number of slices
     * @param rows    number of rows
     * @param columns number of columns
     *  
     */
    public DoubleFFT_3D(long slices, long rows, long columns)
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
        CommonUtils.setUseLargeArrays(2 * slices * rows * columns > LargeArray.getMaxSizeOf32bitArray());
        fftSlices = new DoubleFFT_1D(slices);
        if (slices == rows) {
            fftRows = fftSlices;
        } else {
            fftRows = new DoubleFFT_1D(rows);
        }
        if (slices == columns) {
            fftColumns = fftSlices;
        } else if (rows == columns) {
            fftColumns = fftRows;
        } else {
            fftColumns = new DoubleFFT_1D(columns);
        }
    }

    /**
     * Computes 3D forward DFT of complex data leaving the result in
     * <code>a</code>. The data is stored in 1D array addressed in slice-major,
     * then row-major, then column-major, in order of significance, i.e. element
     * (i,j,k) of 3D array x[slices][rows][2*columns] is stored in
     * a[i*sliceStride + j*rowStride + k], where sliceStride = rows * 2 *
     * columns and rowStride = 2 * columns. Complex number is stored as two
     * double values in sequence: the real and imaginary part, i.e. the input
     * array must be of size slices*rows*2*columns. The physical layout of the
     * input data is as follows:
     *  
     * <pre>
     * a[k1*sliceStride + k2*rowStride + 2*k3] = Re[k1][k2][k3],
     * a[k1*sliceStride + k2*rowStride + 2*k3+1] = Im[k1][k2][k3],
     * 0&lt;=k1&lt;slices, 0&lt;=k2&lt;rows, 0&lt;=k3&lt;columns,
     * </pre>
     *  
     * @param a data to transform
     */
    public void complexForward(final double[] a)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            int oldn3 = columns;
            columns = 2 * columns;
            sliceStride = rows * columns;
            rowStride = columns;
            if ((nthreads > 1) && useThreads) {
                xdft3da_subth2(0, -1, a, true);
                cdft3db_subth(-1, a, true);
            } else {
                xdft3da_sub2(0, -1, a, true);
                cdft3db_sub(-1, a, true);
            }
            columns = oldn3;
            sliceStride = rows * columns;
            rowStride = columns;
        } else {
            sliceStride = 2 * rows * columns;
            rowStride = 2 * columns;
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
                                    fftColumns.complexForward(a, idx1 + r * rowStride);
                                }
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
                }

                for (int l = 0; l < nthreads; l++) {
                    final int firstSlice = l * p;
                    final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;

                    futures[l] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            double[] temp = new double[2 * rows];
                            for (int s = firstSlice; s < lastSlice; s++) {
                                int idx1 = s * sliceStride;
                                for (int c = 0; c < columns; c++) {
                                    int idx2 = 2 * c;
                                    for (int r = 0; r < rows; r++) {
                                        int idx3 = idx1 + idx2 + r * rowStride;
                                        int idx4 = 2 * r;
                                        temp[idx4] = a[idx3];
                                        temp[idx4 + 1] = a[idx3 + 1];
                                    }
                                    fftRows.complexForward(temp);
                                    for (int r = 0; r < rows; r++) {
                                        int idx3 = idx1 + idx2 + r * rowStride;
                                        int idx4 = 2 * r;
                                        a[idx3] = temp[idx4];
                                        a[idx3 + 1] = temp[idx4 + 1];
                                    }
                                }
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
                }

                p = rows / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final int firstRow = l * p;
                    final int lastRow = (l == (nthreads - 1)) ? rows : firstRow + p;

                    futures[l] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            double[] temp = new double[2 * slices];
                            for (int r = firstRow; r < lastRow; r++) {
                                int idx1 = r * rowStride;
                                for (int c = 0; c < columns; c++) {
                                    int idx2 = 2 * c;
                                    for (int s = 0; s < slices; s++) {
                                        int idx3 = s * sliceStride + idx1 + idx2;
                                        int idx4 = 2 * s;
                                        temp[idx4] = a[idx3];
                                        temp[idx4 + 1] = a[idx3 + 1];
                                    }
                                    fftSlices.complexForward(temp);
                                    for (int s = 0; s < slices; s++) {
                                        int idx3 = s * sliceStride + idx1 + idx2;
                                        int idx4 = 2 * s;
                                        a[idx3] = temp[idx4];
                                        a[idx3 + 1] = temp[idx4 + 1];
                                    }
                                }
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
                }

            } else {
                for (int s = 0; s < slices; s++) {
                    int idx1 = s * sliceStride;
                    for (int r = 0; r < rows; r++) {
                        fftColumns.complexForward(a, idx1 + r * rowStride);
                    }
                }

                double[] temp = new double[2 * rows];
                for (int s = 0; s < slices; s++) {
                    int idx1 = s * sliceStride;
                    for (int c = 0; c < columns; c++) {
                        int idx2 = 2 * c;
                        for (int r = 0; r < rows; r++) {
                            int idx3 = idx1 + idx2 + r * rowStride;
                            int idx4 = 2 * r;
                            temp[idx4] = a[idx3];
                            temp[idx4 + 1] = a[idx3 + 1];
                        }
                        fftRows.complexForward(temp);
                        for (int r = 0; r < rows; r++) {
                            int idx3 = idx1 + idx2 + r * rowStride;
                            int idx4 = 2 * r;
                            a[idx3] = temp[idx4];
                            a[idx3 + 1] = temp[idx4 + 1];
                        }
                    }
                }

                temp = new double[2 * slices];
                for (int r = 0; r < rows; r++) {
                    int idx1 = r * rowStride;
                    for (int c = 0; c < columns; c++) {
                        int idx2 = 2 * c;
                        for (int s = 0; s < slices; s++) {
                            int idx3 = s * sliceStride + idx1 + idx2;
                            int idx4 = 2 * s;
                            temp[idx4] = a[idx3];
                            temp[idx4 + 1] = a[idx3 + 1];
                        }
                        fftSlices.complexForward(temp);
                        for (int s = 0; s < slices; s++) {
                            int idx3 = s * sliceStride + idx1 + idx2;
                            int idx4 = 2 * s;
                            a[idx3] = temp[idx4];
                            a[idx3 + 1] = temp[idx4 + 1];
                        }
                    }
                }
            }
            sliceStride = rows * columns;
            rowStride = columns;
        }
    }

    /**
     * Computes 3D forward DFT of complex data leaving the result in
     * <code>a</code>. The data is stored in 1D array addressed in slice-major,
     * then row-major, then column-major, in order of significance, i.e. element
     * (i,j,k) of 3D array x[slices][rows][2*columns] is stored in
     * a[i*sliceStride + j*rowStride + k], where sliceStride = rows * 2 *
     * columns and rowStride = 2 * columns. Complex number is stored as two
     * double values in sequence: the real and imaginary part, i.e. the input
     * array must be of size slices*rows*2*columns. The physical layout of the
     * input data is as follows:
     *  
     * <pre>
     * a[k1*sliceStride + k2*rowStride + 2*k3] = Re[k1][k2][k3],
     * a[k1*sliceStride + k2*rowStride + 2*k3+1] = Im[k1][k2][k3],
     * 0&lt;=k1&lt;slices, 0&lt;=k2&lt;rows, 0&lt;=k3&lt;columns,
     * </pre>
     *  
     * @param a data to transform
     */
    public void complexForward(final DoubleLargeArray a)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            long oldn3 = columnsl;
            columnsl = 2 * columnsl;
            sliceStridel = rowsl * columnsl;
            rowStridel = columnsl;
            if ((nthreads > 1) && useThreads) {
                xdft3da_subth2(0, -1, a, true);
                cdft3db_subth(-1, a, true);
            } else {
                xdft3da_sub2(0, -1, a, true);
                cdft3db_sub(-1, a, true);
            }
            columnsl = oldn3;
            sliceStridel = rowsl * columnsl;
            rowStridel = columnsl;
        } else {
            sliceStridel = 2 * rowsl * columnsl;
            rowStridel = 2 * columnsl;
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
                                    fftColumns.complexForward(a, idx1 + r * rowStridel);
                                }
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
                }

                for (int l = 0; l < nthreads; l++) {
                    final long firstSlice = l * p;
                    final long lastSlice = (l == (nthreads - 1)) ? slicesl : firstSlice + p;

                    futures[l] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            DoubleLargeArray temp = new DoubleLargeArray(2 * rowsl, false);
                            for (long s = firstSlice; s < lastSlice; s++) {
                                long idx1 = s * sliceStridel;
                                for (long c = 0; c < columnsl; c++) {
                                    long idx2 = 2 * c;
                                    for (long r = 0; r < rowsl; r++) {
                                        long idx3 = idx1 + idx2 + r * rowStridel;
                                        long idx4 = 2 * r;
                                        temp.setDouble(idx4, a.getDouble(idx3));
                                        temp.setDouble(idx4 + 1, a.getDouble(idx3 + 1));
                                    }
                                    fftRows.complexForward(temp);
                                    for (long r = 0; r < rowsl; r++) {
                                        long idx3 = idx1 + idx2 + r * rowStridel;
                                        long idx4 = 2 * r;
                                        a.setDouble(idx3, temp.getDouble(idx4));
                                        a.setDouble(idx3 + 1, temp.getDouble(idx4 + 1));
                                    }
                                }
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
                }

                p = rowsl / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final long firstRow = l * p;
                    final long lastRow = (l == (nthreads - 1)) ? rowsl : firstRow + p;

                    futures[l] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            DoubleLargeArray temp = new DoubleLargeArray(2 * slicesl, false);
                            for (long r = firstRow; r < lastRow; r++) {
                                long idx1 = r * rowStridel;
                                for (long c = 0; c < columnsl; c++) {
                                    long idx2 = 2 * c;
                                    for (long s = 0; s < slicesl; s++) {
                                        long idx3 = s * sliceStridel + idx1 + idx2;
                                        long idx4 = 2 * s;
                                        temp.setDouble(idx4, a.getDouble(idx3));
                                        temp.setDouble(idx4 + 1, a.getDouble(idx3 + 1));
                                    }
                                    fftSlices.complexForward(temp);
                                    for (long s = 0; s < slicesl; s++) {
                                        long idx3 = s * sliceStridel + idx1 + idx2;
                                        long idx4 = 2 * s;
                                        a.setDouble(idx3, temp.getDouble(idx4));
                                        a.setDouble(idx3 + 1, temp.getDouble(idx4 + 1));
                                    }
                                }
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
                }

            } else {
                for (long s = 0; s < slicesl; s++) {
                    long idx1 = s * sliceStridel;
                    for (long r = 0; r < rowsl; r++) {
                        fftColumns.complexForward(a, idx1 + r * rowStridel);
                    }
                }

                DoubleLargeArray temp = new DoubleLargeArray(2 * rowsl, false);
                for (long s = 0; s < slicesl; s++) {
                    long idx1 = s * sliceStridel;
                    for (long c = 0; c < columnsl; c++) {
                        long idx2 = 2 * c;
                        for (long r = 0; r < rowsl; r++) {
                            long idx3 = idx1 + idx2 + r * rowStridel;
                            long idx4 = 2 * r;
                            temp.setDouble(idx4, a.getDouble(idx3));
                            temp.setDouble(idx4 + 1, a.getDouble(idx3 + 1));
                        }
                        fftRows.complexForward(temp);
                        for (long r = 0; r < rowsl; r++) {
                            long idx3 = idx1 + idx2 + r * rowStridel;
                            long idx4 = 2 * r;
                            a.setDouble(idx3, temp.getDouble(idx4));
                            a.setDouble(idx3 + 1, temp.getDouble(idx4 + 1));
                        }
                    }
                }

                temp = new DoubleLargeArray(2 * slicesl, false);
                for (long r = 0; r < rowsl; r++) {
                    long idx1 = r * rowStridel;
                    for (long c = 0; c < columnsl; c++) {
                        long idx2 = 2 * c;
                        for (long s = 0; s < slicesl; s++) {
                            long idx3 = s * sliceStridel + idx1 + idx2;
                            long idx4 = 2 * s;
                            temp.setDouble(idx4, a.getDouble(idx3));
                            temp.setDouble(idx4 + 1, a.getDouble(idx3 + 1));
                        }
                        fftSlices.complexForward(temp);
                        for (long s = 0; s < slicesl; s++) {
                            long idx3 = s * sliceStridel + idx1 + idx2;
                            long idx4 = 2 * s;
                            a.setDouble(idx3, temp.getDouble(idx4));
                            a.setDouble(idx3 + 1, temp.getDouble(idx4 + 1));
                        }
                    }
                }
            }
            sliceStridel = rowsl * columnsl;
            rowStridel = columnsl;
        }
    }

    /**
     * Computes 3D forward DFT of complex data leaving the result in
     * <code>a</code>. The data is stored in 3D array. Complex data is
     * represented by 2 double values in sequence: the real and imaginary part,
     * i.e. the input array must be of size slices by rows by 2*columns. The
     * physical layout of the input data is as follows:
     *  
     * <pre>
     * a[k1][k2][2*k3] = Re[k1][k2][k3], a[k1][k2][2*k3+1] = Im[k1][k2][k3],
     * 0&lt;=k1&lt;slices, 0&lt;=k2&lt;rows, 0&lt;=k3&lt;columns,
     * </pre>
     *  
     * @param a data to transform
     */
    public void complexForward(final double[][][] a)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            int oldn3 = columns;
            columns = 2 * columns;

            sliceStride = rows * columns;
            rowStride = columns;

            if ((nthreads > 1) && useThreads) {
                xdft3da_subth2(0, -1, a, true);
                cdft3db_subth(-1, a, true);
            } else {
                xdft3da_sub2(0, -1, a, true);
                cdft3db_sub(-1, a, true);
            }
            columns = oldn3;
            sliceStride = rows * columns;
            rowStride = columns;
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
                                fftColumns.complexForward(a[s][r]);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;

                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        double[] temp = new double[2 * rows];
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int c = 0; c < columns; c++) {
                                int idx2 = 2 * c;
                                for (int r = 0; r < rows; r++) {
                                    int idx4 = 2 * r;
                                    temp[idx4] = a[s][r][idx2];
                                    temp[idx4 + 1] = a[s][r][idx2 + 1];
                                }
                                fftRows.complexForward(temp);
                                for (int r = 0; r < rows; r++) {
                                    int idx4 = 2 * r;
                                    a[s][r][idx2] = temp[idx4];
                                    a[s][r][idx2 + 1] = temp[idx4 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            p = rows / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstRow = l * p;
                final int lastRow = (l == (nthreads - 1)) ? rows : firstRow + p;

                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        double[] temp = new double[2 * slices];
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                int idx2 = 2 * c;
                                for (int s = 0; s < slices; s++) {
                                    int idx4 = 2 * s;
                                    temp[idx4] = a[s][r][idx2];
                                    temp[idx4 + 1] = a[s][r][idx2 + 1];
                                }
                                fftSlices.complexForward(temp);
                                for (int s = 0; s < slices; s++) {
                                    int idx4 = 2 * s;
                                    a[s][r][idx2] = temp[idx4];
                                    a[s][r][idx2 + 1] = temp[idx4 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

        } else {
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    fftColumns.complexForward(a[s][r]);
                }
            }

            double[] temp = new double[2 * rows];
            for (int s = 0; s < slices; s++) {
                for (int c = 0; c < columns; c++) {
                    int idx2 = 2 * c;
                    for (int r = 0; r < rows; r++) {
                        int idx4 = 2 * r;
                        temp[idx4] = a[s][r][idx2];
                        temp[idx4 + 1] = a[s][r][idx2 + 1];
                    }
                    fftRows.complexForward(temp);
                    for (int r = 0; r < rows; r++) {
                        int idx4 = 2 * r;
                        a[s][r][idx2] = temp[idx4];
                        a[s][r][idx2 + 1] = temp[idx4 + 1];
                    }
                }
            }

            temp = new double[2 * slices];
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    int idx2 = 2 * c;
                    for (int s = 0; s < slices; s++) {
                        int idx4 = 2 * s;
                        temp[idx4] = a[s][r][idx2];
                        temp[idx4 + 1] = a[s][r][idx2 + 1];
                    }
                    fftSlices.complexForward(temp);
                    for (int s = 0; s < slices; s++) {
                        int idx4 = 2 * s;
                        a[s][r][idx2] = temp[idx4];
                        a[s][r][idx2 + 1] = temp[idx4 + 1];
                    }
                }
            }
        }
    }

    /**
     * Computes 3D inverse DFT of complex data leaving the result in
     * <code>a</code>. The data is stored in a 1D array addressed in
     * slice-major, then row-major, then column-major, in order of significance,
     * i.e. element (i,j,k) of 3-d array x[slices][rows][2*columns] is stored in
     * a[i*sliceStride + j*rowStride + k], where sliceStride = rows * 2 *
     * columns and rowStride = 2 * columns. Complex number is stored as two
     * double values in sequence: the real and imaginary part, i.e. the input
     * array must be of size slices*rows*2*columns. The physical layout of the
     * input data is as follows:
     *  
     * <pre>
     * a[k1*sliceStride + k2*rowStride + 2*k3] = Re[k1][k2][k3],
     * a[k1*sliceStride + k2*rowStride + 2*k3+1] = Im[k1][k2][k3],
     * 0&lt;=k1&lt;slices, 0&lt;=k2&lt;rows, 0&lt;=k3&lt;columns,
     * </pre>
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void complexInverse(final double[] a, final boolean scale)
    {

        int nthreads = ConcurrencyUtils.getNumberOfThreads();

        if (isPowerOfTwo) {
            int oldn3 = columns;
            columns = 2 * columns;
            sliceStride = rows * columns;
            rowStride = columns;
            if ((nthreads > 1) && useThreads) {
                xdft3da_subth2(0, 1, a, scale);
                cdft3db_subth(1, a, scale);
            } else {
                xdft3da_sub2(0, 1, a, scale);
                cdft3db_sub(1, a, scale);
            }
            columns = oldn3;
            sliceStride = rows * columns;
            rowStride = columns;
        } else {
            sliceStride = 2 * rows * columns;
            rowStride = 2 * columns;
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
                                    fftColumns.complexInverse(a, idx1 + r * rowStride, scale);
                                }
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
                }

                for (int l = 0; l < nthreads; l++) {
                    final int firstSlice = l * p;
                    final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;

                    futures[l] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            double[] temp = new double[2 * rows];
                            for (int s = firstSlice; s < lastSlice; s++) {
                                int idx1 = s * sliceStride;
                                for (int c = 0; c < columns; c++) {
                                    int idx2 = 2 * c;
                                    for (int r = 0; r < rows; r++) {
                                        int idx3 = idx1 + idx2 + r * rowStride;
                                        int idx4 = 2 * r;
                                        temp[idx4] = a[idx3];
                                        temp[idx4 + 1] = a[idx3 + 1];
                                    }
                                    fftRows.complexInverse(temp, scale);
                                    for (int r = 0; r < rows; r++) {
                                        int idx3 = idx1 + idx2 + r * rowStride;
                                        int idx4 = 2 * r;
                                        a[idx3] = temp[idx4];
                                        a[idx3 + 1] = temp[idx4 + 1];
                                    }
                                }
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
                }

                p = rows / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final int firstRow = l * p;
                    final int lastRow = (l == (nthreads - 1)) ? rows : firstRow + p;

                    futures[l] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            double[] temp = new double[2 * slices];
                            for (int r = firstRow; r < lastRow; r++) {
                                int idx1 = r * rowStride;
                                for (int c = 0; c < columns; c++) {
                                    int idx2 = 2 * c;
                                    for (int s = 0; s < slices; s++) {
                                        int idx3 = s * sliceStride + idx1 + idx2;
                                        int idx4 = 2 * s;
                                        temp[idx4] = a[idx3];
                                        temp[idx4 + 1] = a[idx3 + 1];
                                    }
                                    fftSlices.complexInverse(temp, scale);
                                    for (int s = 0; s < slices; s++) {
                                        int idx3 = s * sliceStride + idx1 + idx2;
                                        int idx4 = 2 * s;
                                        a[idx3] = temp[idx4];
                                        a[idx3 + 1] = temp[idx4 + 1];
                                    }
                                }
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
                }

            } else {
                for (int s = 0; s < slices; s++) {
                    int idx1 = s * sliceStride;
                    for (int r = 0; r < rows; r++) {
                        fftColumns.complexInverse(a, idx1 + r * rowStride, scale);
                    }
                }
                double[] temp = new double[2 * rows];
                for (int s = 0; s < slices; s++) {
                    int idx1 = s * sliceStride;
                    for (int c = 0; c < columns; c++) {
                        int idx2 = 2 * c;
                        for (int r = 0; r < rows; r++) {
                            int idx3 = idx1 + idx2 + r * rowStride;
                            int idx4 = 2 * r;
                            temp[idx4] = a[idx3];
                            temp[idx4 + 1] = a[idx3 + 1];
                        }
                        fftRows.complexInverse(temp, scale);
                        for (int r = 0; r < rows; r++) {
                            int idx3 = idx1 + idx2 + r * rowStride;
                            int idx4 = 2 * r;
                            a[idx3] = temp[idx4];
                            a[idx3 + 1] = temp[idx4 + 1];
                        }
                    }
                }
                temp = new double[2 * slices];
                for (int r = 0; r < rows; r++) {
                    int idx1 = r * rowStride;
                    for (int c = 0; c < columns; c++) {
                        int idx2 = 2 * c;
                        for (int s = 0; s < slices; s++) {
                            int idx3 = s * sliceStride + idx1 + idx2;
                            int idx4 = 2 * s;
                            temp[idx4] = a[idx3];
                            temp[idx4 + 1] = a[idx3 + 1];
                        }
                        fftSlices.complexInverse(temp, scale);
                        for (int s = 0; s < slices; s++) {
                            int idx3 = s * sliceStride + idx1 + idx2;
                            int idx4 = 2 * s;
                            a[idx3] = temp[idx4];
                            a[idx3 + 1] = temp[idx4 + 1];
                        }
                    }
                }
            }
            sliceStride = rows * columns;
            rowStride = columns;
        }
    }

    /**
     * Computes 3D inverse DFT of complex data leaving the result in
     * <code>a</code>. The data is stored in a 1D array addressed in
     * slice-major, then row-major, then column-major, in order of significance,
     * i.e. element (i,j,k) of 3-d array x[slices][rows][2*columns] is stored in
     * a[i*sliceStride + j*rowStride + k], where sliceStride = rows * 2 *
     * columns and rowStride = 2 * columns. Complex number is stored as two
     * double values in sequence: the real and imaginary part, i.e. the input
     * array must be of size slices*rows*2*columns. The physical layout of the
     * input data is as follows:
     *  
     * <pre>
     * a[k1*sliceStride + k2*rowStride + 2*k3] = Re[k1][k2][k3],
     * a[k1*sliceStride + k2*rowStride + 2*k3+1] = Im[k1][k2][k3],
     * 0&lt;=k1&lt;slices, 0&lt;=k2&lt;rows, 0&lt;=k3&lt;columns,
     * </pre>
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void complexInverse(final DoubleLargeArray a, final boolean scale)
    {

        int nthreads = ConcurrencyUtils.getNumberOfThreads();

        if (isPowerOfTwo) {
            long oldn3 = columnsl;
            columnsl = 2 * columnsl;
            sliceStridel = rowsl * columnsl;
            rowStridel = columnsl;
            if ((nthreads > 1) && useThreads) {
                xdft3da_subth2(0, 1, a, scale);
                cdft3db_subth(1, a, scale);
            } else {
                xdft3da_sub2(0, 1, a, scale);
                cdft3db_sub(1, a, scale);
            }
            columnsl = oldn3;
            sliceStridel = rowsl * columnsl;
            rowStridel = columnsl;
        } else {
            sliceStridel = 2 * rowsl * columnsl;
            rowStridel = 2 * columnsl;
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
                                    fftColumns.complexInverse(a, idx1 + r * rowStridel, scale);
                                }
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
                }

                for (int l = 0; l < nthreads; l++) {
                    final long firstSlice = l * p;
                    final long lastSlice = (l == (nthreads - 1)) ? slicesl : firstSlice + p;

                    futures[l] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            DoubleLargeArray temp = new DoubleLargeArray(2 * rowsl, false);
                            for (long s = firstSlice; s < lastSlice; s++) {
                                long idx1 = s * sliceStridel;
                                for (long c = 0; c < columnsl; c++) {
                                    long idx2 = 2 * c;
                                    for (long r = 0; r < rowsl; r++) {
                                        long idx3 = idx1 + idx2 + r * rowStridel;
                                        long idx4 = 2 * r;
                                        temp.setDouble(idx4, a.getDouble(idx3));
                                        temp.setDouble(idx4 + 1, a.getDouble(idx3 + 1));
                                    }
                                    fftRows.complexInverse(temp, scale);
                                    for (long r = 0; r < rowsl; r++) {
                                        long idx3 = idx1 + idx2 + r * rowStridel;
                                        long idx4 = 2 * r;
                                        a.setDouble(idx3, temp.getDouble(idx4));
                                        a.setDouble(idx3 + 1, temp.getDouble(idx4 + 1));
                                    }
                                }
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
                }

                p = rowsl / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final long firstRow = l * p;
                    final long lastRow = (l == (nthreads - 1)) ? rowsl : firstRow + p;

                    futures[l] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            DoubleLargeArray temp = new DoubleLargeArray(2 * slicesl, false);
                            for (long r = firstRow; r < lastRow; r++) {
                                long idx1 = r * rowStridel;
                                for (long c = 0; c < columnsl; c++) {
                                    long idx2 = 2 * c;
                                    for (long s = 0; s < slicesl; s++) {
                                        long idx3 = s * sliceStridel + idx1 + idx2;
                                        long idx4 = 2 * s;
                                        temp.setDouble(idx4, a.getDouble(idx3));
                                        temp.setDouble(idx4 + 1, a.getDouble(idx3 + 1));
                                    }
                                    fftSlices.complexInverse(temp, scale);
                                    for (long s = 0; s < slicesl; s++) {
                                        long idx3 = s * sliceStridel + idx1 + idx2;
                                        long idx4 = 2 * s;
                                        a.setDouble(idx3, temp.getDouble(idx4));
                                        a.setDouble(idx3 + 1, temp.getDouble(idx4 + 1));
                                    }
                                }
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
                }

            } else {
                for (long s = 0; s < slicesl; s++) {
                    long idx1 = s * sliceStridel;
                    for (long r = 0; r < rowsl; r++) {
                        fftColumns.complexInverse(a, idx1 + r * rowStridel, scale);
                    }
                }
                DoubleLargeArray temp = new DoubleLargeArray(2 * rowsl, false);
                for (long s = 0; s < slicesl; s++) {
                    long idx1 = s * sliceStridel;
                    for (long c = 0; c < columnsl; c++) {
                        long idx2 = 2 * c;
                        for (long r = 0; r < rowsl; r++) {
                            long idx3 = idx1 + idx2 + r * rowStridel;
                            long idx4 = 2 * r;
                            temp.setDouble(idx4, a.getDouble(idx3));
                            temp.setDouble(idx4 + 1, a.getDouble(idx3 + 1));
                        }
                        fftRows.complexInverse(temp, scale);
                        for (long r = 0; r < rowsl; r++) {
                            long idx3 = idx1 + idx2 + r * rowStridel;
                            long idx4 = 2 * r;
                            a.setDouble(idx3, temp.getDouble(idx4));
                            a.setDouble(idx3 + 1, temp.getDouble(idx4 + 1));
                        }
                    }
                }
                temp = new DoubleLargeArray(2 * slicesl, false);
                for (long r = 0; r < rowsl; r++) {
                    long idx1 = r * rowStridel;
                    for (long c = 0; c < columnsl; c++) {
                        long idx2 = 2 * c;
                        for (long s = 0; s < slicesl; s++) {
                            long idx3 = s * sliceStridel + idx1 + idx2;
                            long idx4 = 2 * s;
                            temp.setDouble(idx4, a.getDouble(idx3));
                            temp.setDouble(idx4 + 1, a.getDouble(idx3 + 1));
                        }
                        fftSlices.complexInverse(temp, scale);
                        for (long s = 0; s < slicesl; s++) {
                            long idx3 = s * sliceStridel + idx1 + idx2;
                            long idx4 = 2 * s;
                            a.setDouble(idx3, temp.getDouble(idx4));
                            a.setDouble(idx3 + 1, temp.getDouble(idx4 + 1));
                        }
                    }
                }
            }
            sliceStridel = rowsl * columnsl;
            rowStridel = columnsl;
        }
    }

    /**
     * Computes 3D inverse DFT of complex data leaving the result in
     * <code>a</code>. The data is stored in a 3D array. Complex data is
     * represented by 2 double values in sequence: the real and imaginary part,
     * i.e. the input array must be of size slices by rows by 2*columns. The
     * physical layout of the input data is as follows:
     *  
     * <pre>
     * a[k1][k2][2*k3] = Re[k1][k2][k3], a[k1][k2][2*k3+1] = Im[k1][k2][k3],
     * 0&lt;=k1&lt;slices, 0&lt;=k2&lt;rows, 0&lt;=k3&lt;columns,
     * </pre>
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void complexInverse(final double[][][] a, final boolean scale)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            int oldn3 = columns;
            columns = 2 * columns;
            sliceStride = rows * columns;
            rowStride = columns;
            if ((nthreads > 1) && useThreads) {
                xdft3da_subth2(0, 1, a, scale);
                cdft3db_subth(1, a, scale);
            } else {
                xdft3da_sub2(0, 1, a, scale);
                cdft3db_sub(1, a, scale);
            }
            columns = oldn3;
            sliceStride = rows * columns;
            rowStride = columns;
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
                                fftColumns.complexInverse(a[s][r], scale);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;

                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        double[] temp = new double[2 * rows];
                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int c = 0; c < columns; c++) {
                                int idx2 = 2 * c;
                                for (int r = 0; r < rows; r++) {
                                    int idx4 = 2 * r;
                                    temp[idx4] = a[s][r][idx2];
                                    temp[idx4 + 1] = a[s][r][idx2 + 1];
                                }
                                fftRows.complexInverse(temp, scale);
                                for (int r = 0; r < rows; r++) {
                                    int idx4 = 2 * r;
                                    a[s][r][idx2] = temp[idx4];
                                    a[s][r][idx2 + 1] = temp[idx4 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            p = rows / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstRow = l * p;
                final int lastRow = (l == (nthreads - 1)) ? rows : firstRow + p;

                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        double[] temp = new double[2 * slices];
                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                int idx2 = 2 * c;
                                for (int s = 0; s < slices; s++) {
                                    int idx4 = 2 * s;
                                    temp[idx4] = a[s][r][idx2];
                                    temp[idx4 + 1] = a[s][r][idx2 + 1];
                                }
                                fftSlices.complexInverse(temp, scale);
                                for (int s = 0; s < slices; s++) {
                                    int idx4 = 2 * s;
                                    a[s][r][idx2] = temp[idx4];
                                    a[s][r][idx2 + 1] = temp[idx4 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

        } else {
            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    fftColumns.complexInverse(a[s][r], scale);
                }
            }
            double[] temp = new double[2 * rows];
            for (int s = 0; s < slices; s++) {
                for (int c = 0; c < columns; c++) {
                    int idx2 = 2 * c;
                    for (int r = 0; r < rows; r++) {
                        int idx4 = 2 * r;
                        temp[idx4] = a[s][r][idx2];
                        temp[idx4 + 1] = a[s][r][idx2 + 1];
                    }
                    fftRows.complexInverse(temp, scale);
                    for (int r = 0; r < rows; r++) {
                        int idx4 = 2 * r;
                        a[s][r][idx2] = temp[idx4];
                        a[s][r][idx2 + 1] = temp[idx4 + 1];
                    }
                }
            }
            temp = new double[2 * slices];
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    int idx2 = 2 * c;
                    for (int s = 0; s < slices; s++) {
                        int idx4 = 2 * s;
                        temp[idx4] = a[s][r][idx2];
                        temp[idx4 + 1] = a[s][r][idx2 + 1];
                    }
                    fftSlices.complexInverse(temp, scale);
                    for (int s = 0; s < slices; s++) {
                        int idx4 = 2 * s;
                        a[s][r][idx2] = temp[idx4];
                        a[s][r][idx2 + 1] = temp[idx4 + 1];
                    }
                }
            }
        }
    }

    /**
     * Computes 3D forward DFT of real data leaving the result in <code>a</code>
     * . This method only works when the sizes of all three dimensions are
     * power-of-two numbers. The data is stored in a 1D array addressed in
     * slice-major, then row-major, then column-major, in order of significance,
     * i.e. element (i,j,k) of 3-d array x[slices][rows][2*columns] is stored in
     * a[i*sliceStride + j*rowStride + k], where sliceStride = rows * 2 *
     * columns and rowStride = 2 * columns. The physical layout of the output
     * data is as follows:
     *  
     * <pre>
     * a[k1*sliceStride + k2*rowStride + 2*k3] = Re[k1][k2][k3] =
     * Re[(slices-k1)%slices][(rows-k2)%rows][columns-k3], a[k1*sliceStride +
     * k2*rowStride + 2*k3+1] = Im[k1][k2][k3] =
     * -Im[(slices-k1)%slices][(rows-k2)%rows][columns-k3], 0&lt;=k1&lt;slices,
     * 0&lt;=k2&lt;rows, 0&lt;k3&lt;columns/2, a[k1*sliceStride + k2*rowStride]
     * = Re[k1][k2][0] = Re[(slices-k1)%slices][rows-k2][0], a[k1*sliceStride +
     * k2*rowStride + 1] = Im[k1][k2][0] = -Im[(slices-k1)%slices][rows-k2][0],
     * a[k1*sliceStride + (rows-k2)*rowStride + 1] =
     * Re[(slices-k1)%slices][k2][columns/2] = Re[k1][rows-k2][columns/2],
     * a[k1*sliceStride + (rows-k2)*rowStride] =
     * -Im[(slices-k1)%slices][k2][columns/2] = Im[k1][rows-k2][columns/2],
     * 0&lt;=k1&lt;slices, 0&lt;k2&lt;rows/2, a[k1*sliceStride] = Re[k1][0][0] =
     * Re[slices-k1][0][0], a[k1*sliceStride + 1] = Im[k1][0][0] =
     * -Im[slices-k1][0][0], a[k1*sliceStride + (rows/2)*rowStride] =
     * Re[k1][rows/2][0] = Re[slices-k1][rows/2][0], a[k1*sliceStride +
     * (rows/2)*rowStride + 1] = Im[k1][rows/2][0] = -Im[slices-k1][rows/2][0],
     * a[(slices-k1)*sliceStride + 1] = Re[k1][0][columns/2] =
     * Re[slices-k1][0][columns/2], a[(slices-k1)*sliceStride] =
     * -Im[k1][0][columns/2] = Im[slices-k1][0][columns/2],
     * a[(slices-k1)*sliceStride + (rows/2)*rowStride + 1] =
     * Re[k1][rows/2][columns/2] = Re[slices-k1][rows/2][columns/2],
     * a[(slices-k1)*sliceStride + (rows/2) * rowStride] =
     * -Im[k1][rows/2][columns/2] = Im[slices-k1][rows/2][columns/2],
     * 0&lt;k1&lt;slices/2, a[0] = Re[0][0][0], a[1] = Re[0][0][columns/2],
     * a[(rows/2)*rowStride] = Re[0][rows/2][0], a[(rows/2)*rowStride + 1] =
     * Re[0][rows/2][columns/2], a[(slices/2)*sliceStride] = Re[slices/2][0][0],
     * a[(slices/2)*sliceStride + 1] = Re[slices/2][0][columns/2],
     * a[(slices/2)*sliceStride + (rows/2)*rowStride] = Re[slices/2][rows/2][0],
     * a[(slices/2)*sliceStride + (rows/2)*rowStride + 1] =
     * Re[slices/2][rows/2][columns/2]
     * </pre>
     *  
     *  
     * This method computes only half of the elements of the real transform. The
     * other half satisfies the symmetry condition. If you want the full real
     * forward transform, use <code>realForwardFull</code>. To get back the
     * original data, use <code>realInverse</code> on the output of this method.
     *  
     * @param a data to transform
     */
    public void realForward(double[] a)
    {
        if (isPowerOfTwo == false) {
            throw new IllegalArgumentException("slices, rows and columns must be power of two numbers");
        } else {
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && useThreads) {
                xdft3da_subth1(1, -1, a, true);
                cdft3db_subth(-1, a, true);
                rdft3d_sub(1, a);
            } else {
                xdft3da_sub1(1, -1, a, true);
                cdft3db_sub(-1, a, true);
                rdft3d_sub(1, a);
            }
        }
    }

    /**
     * Computes 3D forward DFT of real data leaving the result in <code>a</code>
     * . This method only works when the sizes of all three dimensions are
     * power-of-two numbers. The data is stored in a 1D array addressed in
     * slice-major, then row-major, then column-major, in order of significance,
     * i.e. element (i,j,k) of 3-d array x[slices][rows][2*columns] is stored in
     * a[i*sliceStride + j*rowStride + k], where sliceStride = rows * 2 *
     * columns and rowStride = 2 * columns. The physical layout of the output
     * data is as follows:
     *  
     * <pre>
     * a[k1*sliceStride + k2*rowStride + 2*k3] = Re[k1][k2][k3] =
     * Re[(slices-k1)%slices][(rows-k2)%rows][columns-k3], a[k1*sliceStride +
     * k2*rowStride + 2*k3+1] = Im[k1][k2][k3] =
     * -Im[(slices-k1)%slices][(rows-k2)%rows][columns-k3], 0&lt;=k1&lt;slices,
     * 0&lt;=k2&lt;rows, 0&lt;k3&lt;columns/2, a[k1*sliceStride + k2*rowStride]
     * = Re[k1][k2][0] = Re[(slices-k1)%slices][rows-k2][0], a[k1*sliceStride +
     * k2*rowStride + 1] = Im[k1][k2][0] = -Im[(slices-k1)%slices][rows-k2][0],
     * a[k1*sliceStride + (rows-k2)*rowStride + 1] =
     * Re[(slices-k1)%slices][k2][columns/2] = Re[k1][rows-k2][columns/2],
     * a[k1*sliceStride + (rows-k2)*rowStride] =
     * -Im[(slices-k1)%slices][k2][columns/2] = Im[k1][rows-k2][columns/2],
     * 0&lt;=k1&lt;slices, 0&lt;k2&lt;rows/2, a[k1*sliceStride] = Re[k1][0][0] =
     * Re[slices-k1][0][0], a[k1*sliceStride + 1] = Im[k1][0][0] =
     * -Im[slices-k1][0][0], a[k1*sliceStride + (rows/2)*rowStride] =
     * Re[k1][rows/2][0] = Re[slices-k1][rows/2][0], a[k1*sliceStride +
     * (rows/2)*rowStride + 1] = Im[k1][rows/2][0] = -Im[slices-k1][rows/2][0],
     * a[(slices-k1)*sliceStride + 1] = Re[k1][0][columns/2] =
     * Re[slices-k1][0][columns/2], a[(slices-k1)*sliceStride] =
     * -Im[k1][0][columns/2] = Im[slices-k1][0][columns/2],
     * a[(slices-k1)*sliceStride + (rows/2)*rowStride + 1] =
     * Re[k1][rows/2][columns/2] = Re[slices-k1][rows/2][columns/2],
     * a[(slices-k1)*sliceStride + (rows/2) * rowStride] =
     * -Im[k1][rows/2][columns/2] = Im[slices-k1][rows/2][columns/2],
     * 0&lt;k1&lt;slices/2, a[0] = Re[0][0][0], a[1] = Re[0][0][columns/2],
     * a[(rows/2)*rowStride] = Re[0][rows/2][0], a[(rows/2)*rowStride + 1] =
     * Re[0][rows/2][columns/2], a[(slices/2)*sliceStride] = Re[slices/2][0][0],
     * a[(slices/2)*sliceStride + 1] = Re[slices/2][0][columns/2],
     * a[(slices/2)*sliceStride + (rows/2)*rowStride] = Re[slices/2][rows/2][0],
     * a[(slices/2)*sliceStride + (rows/2)*rowStride + 1] =
     * Re[slices/2][rows/2][columns/2]
     * </pre>
     *  
     *  
     * This method computes only half of the elements of the real transform. The
     * other half satisfies the symmetry condition. If you want the full real
     * forward transform, use <code>realForwardFull</code>. To get back the
     * original data, use <code>realInverse</code> on the output of this method.
     *  
     * @param a data to transform
     */
    public void realForward(DoubleLargeArray a)
    {
        if (isPowerOfTwo == false) {
            throw new IllegalArgumentException("slices, rows and columns must be power of two numbers");
        } else {
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && useThreads) {
                xdft3da_subth1(1, -1, a, true);
                cdft3db_subth(-1, a, true);
                rdft3d_sub(1, a);
            } else {
                xdft3da_sub1(1, -1, a, true);
                cdft3db_sub(-1, a, true);
                rdft3d_sub(1, a);
            }
        }
    }

    /**
     * Computes 3D forward DFT of real data leaving the result in <code>a</code>
     * . This method only works when the sizes of all three dimensions are
     * power-of-two numbers. The data is stored in a 3D array. The physical
     * layout of the output data is as follows:
     *  
     * <pre>
     * a[k1][k2][2*k3] = Re[k1][k2][k3] =
     * Re[(slices-k1)%slices][(rows-k2)%rows][columns-k3], a[k1][k2][2*k3+1] =
     * Im[k1][k2][k3] = -Im[(slices-k1)%slices][(rows-k2)%rows][columns-k3],
     * 0&lt;=k1&lt;slices, 0&lt;=k2&lt;rows, 0&lt;k3&lt;columns/2, a[k1][k2][0]
     * = Re[k1][k2][0] = Re[(slices-k1)%slices][rows-k2][0], a[k1][k2][1] =
     * Im[k1][k2][0] = -Im[(slices-k1)%slices][rows-k2][0], a[k1][rows-k2][1] =
     * Re[(slices-k1)%slices][k2][columns/2] = Re[k1][rows-k2][columns/2],
     * a[k1][rows-k2][0] = -Im[(slices-k1)%slices][k2][columns/2] =
     * Im[k1][rows-k2][columns/2], 0&lt;=k1&lt;slices, 0&lt;k2&lt;rows/2,
     * a[k1][0][0] = Re[k1][0][0] = Re[slices-k1][0][0], a[k1][0][1] =
     * Im[k1][0][0] = -Im[slices-k1][0][0], a[k1][rows/2][0] = Re[k1][rows/2][0]
     * = Re[slices-k1][rows/2][0], a[k1][rows/2][1] = Im[k1][rows/2][0] =
     * -Im[slices-k1][rows/2][0], a[slices-k1][0][1] = Re[k1][0][columns/2] =
     * Re[slices-k1][0][columns/2], a[slices-k1][0][0] = -Im[k1][0][columns/2] =
     * Im[slices-k1][0][columns/2], a[slices-k1][rows/2][1] =
     * Re[k1][rows/2][columns/2] = Re[slices-k1][rows/2][columns/2],
     * a[slices-k1][rows/2][0] = -Im[k1][rows/2][columns/2] =
     * Im[slices-k1][rows/2][columns/2], 0&lt;k1&lt;slices/2, a[0][0][0] =
     * Re[0][0][0], a[0][0][1] = Re[0][0][columns/2], a[0][rows/2][0] =
     * Re[0][rows/2][0], a[0][rows/2][1] = Re[0][rows/2][columns/2],
     * a[slices/2][0][0] = Re[slices/2][0][0], a[slices/2][0][1] =
     * Re[slices/2][0][columns/2], a[slices/2][rows/2][0] =
     * Re[slices/2][rows/2][0], a[slices/2][rows/2][1] =
     * Re[slices/2][rows/2][columns/2]
     * </pre>
     *  
     *  
     * This method computes only half of the elements of the real transform. The
     * other half satisfies the symmetry condition. If you want the full real
     * forward transform, use <code>realForwardFull</code>. To get back the
     * original data, use <code>realInverse</code> on the output of this method.
     *  
     * @param a data to transform
     */
    public void realForward(double[][][] a)
    {
        if (isPowerOfTwo == false) {
            throw new IllegalArgumentException("slices, rows and columns must be power of two numbers");
        } else {
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && useThreads) {
                xdft3da_subth1(1, -1, a, true);
                cdft3db_subth(-1, a, true);
                rdft3d_sub(1, a);
            } else {
                xdft3da_sub1(1, -1, a, true);
                cdft3db_sub(-1, a, true);
                rdft3d_sub(1, a);
            }
        }
    }

    /**
     * Computes 3D forward DFT of real data leaving the result in <code>a</code>
     * . This method computes full real forward transform, i.e. you will get the
     * same result as from <code>complexForward</code> called with all imaginary
     * part equal 0. Because the result is stored in <code>a</code>, the input
     * array must be of size slices*rows*2*columns, with only the first
     * slices*rows*columns elements filled with real data. To get back the
     * original data, use <code>complexInverse</code> on the output of this
     * method.
     *  
     * @param a data to transform
     */
    public void realForwardFull(double[] a)
    {
        if (isPowerOfTwo) {
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && useThreads) {
                xdft3da_subth2(1, -1, a, true);
                cdft3db_subth(-1, a, true);
                rdft3d_sub(1, a);
            } else {
                xdft3da_sub2(1, -1, a, true);
                cdft3db_sub(-1, a, true);
                rdft3d_sub(1, a);
            }
            fillSymmetric(a);
        } else {
            mixedRadixRealForwardFull(a);
        }
    }

    /**
     * Computes 3D forward DFT of real data leaving the result in <code>a</code>
     * . This method computes full real forward transform, i.e. you will get the
     * same result as from <code>complexForward</code> called with all imaginary
     * part equal 0. Because the result is stored in <code>a</code>, the input
     * array must be of size slices*rows*2*columns, with only the first
     * slices*rows*columns elements filled with real data. To get back the
     * original data, use <code>complexInverse</code> on the output of this
     * method.
     *  
     * @param a data to transform
     */
    public void realForwardFull(DoubleLargeArray a)
    {
        if (isPowerOfTwo) {
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && useThreads) {
                xdft3da_subth2(1, -1, a, true);
                cdft3db_subth(-1, a, true);
                rdft3d_sub(1, a);
            } else {
                xdft3da_sub2(1, -1, a, true);
                cdft3db_sub(-1, a, true);
                rdft3d_sub(1, a);
            }
            fillSymmetric(a);
        } else {
            mixedRadixRealForwardFull(a);
        }
    }

    /**
     * Computes 3D forward DFT of real data leaving the result in <code>a</code>
     * . This method computes full real forward transform, i.e. you will get the
     * same result as from <code>complexForward</code> called with all imaginary
     * part equal 0. Because the result is stored in <code>a</code>, the input
     * array must be of size slices by rows by 2*columns, with only the first
     * slices by rows by columns elements filled with real data. To get back the
     * original data, use <code>complexInverse</code> on the output of this
     * method.
     *  
     * @param a data to transform
     */
    public void realForwardFull(double[][][] a)
    {
        if (isPowerOfTwo) {
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && useThreads) {
                xdft3da_subth2(1, -1, a, true);
                cdft3db_subth(-1, a, true);
                rdft3d_sub(1, a);
            } else {
                xdft3da_sub2(1, -1, a, true);
                cdft3db_sub(-1, a, true);
                rdft3d_sub(1, a);
            }
            fillSymmetric(a);
        } else {
            mixedRadixRealForwardFull(a);
        }
    }

    /**
     * Computes 3D inverse DFT of real data leaving the result in <code>a</code>
     * . This method only works when the sizes of all three dimensions are
     * power-of-two numbers. The data is stored in a 1D array addressed in
     * slice-major, then row-major, then column-major, in order of significance,
     * i.e. element (i,j,k) of 3-d array x[slices][rows][2*columns] is stored in
     * a[i*sliceStride + j*rowStride + k], where sliceStride = rows * 2 *
     * columns and rowStride = 2 * columns. The physical layout of the input
     * data has to be as follows:
     *  
     * <pre>
     * a[k1*sliceStride + k2*rowStride + 2*k3] = Re[k1][k2][k3] =
     * Re[(slices-k1)%slices][(rows-k2)%rows][columns-k3], a[k1*sliceStride +
     * k2*rowStride + 2*k3+1] = Im[k1][k2][k3] =
     * -Im[(slices-k1)%slices][(rows-k2)%rows][columns-k3], 0&lt;=k1&lt;slices,
     * 0&lt;=k2&lt;rows, 0&lt;k3&lt;columns/2, a[k1*sliceStride + k2*rowStride]
     * = Re[k1][k2][0] = Re[(slices-k1)%slices][rows-k2][0], a[k1*sliceStride +
     * k2*rowStride + 1] = Im[k1][k2][0] = -Im[(slices-k1)%slices][rows-k2][0],
     * a[k1*sliceStride + (rows-k2)*rowStride + 1] =
     * Re[(slices-k1)%slices][k2][columns/2] = Re[k1][rows-k2][columns/2],
     * a[k1*sliceStride + (rows-k2)*rowStride] =
     * -Im[(slices-k1)%slices][k2][columns/2] = Im[k1][rows-k2][columns/2],
     * 0&lt;=k1&lt;slices, 0&lt;k2&lt;rows/2, a[k1*sliceStride] = Re[k1][0][0] =
     * Re[slices-k1][0][0], a[k1*sliceStride + 1] = Im[k1][0][0] =
     * -Im[slices-k1][0][0], a[k1*sliceStride + (rows/2)*rowStride] =
     * Re[k1][rows/2][0] = Re[slices-k1][rows/2][0], a[k1*sliceStride +
     * (rows/2)*rowStride + 1] = Im[k1][rows/2][0] = -Im[slices-k1][rows/2][0],
     * a[(slices-k1)*sliceStride + 1] = Re[k1][0][columns/2] =
     * Re[slices-k1][0][columns/2], a[(slices-k1)*sliceStride] =
     * -Im[k1][0][columns/2] = Im[slices-k1][0][columns/2],
     * a[(slices-k1)*sliceStride + (rows/2)*rowStride + 1] =
     * Re[k1][rows/2][columns/2] = Re[slices-k1][rows/2][columns/2],
     * a[(slices-k1)*sliceStride + (rows/2) * rowStride] =
     * -Im[k1][rows/2][columns/2] = Im[slices-k1][rows/2][columns/2],
     * 0&lt;k1&lt;slices/2, a[0] = Re[0][0][0], a[1] = Re[0][0][columns/2],
     * a[(rows/2)*rowStride] = Re[0][rows/2][0], a[(rows/2)*rowStride + 1] =
     * Re[0][rows/2][columns/2], a[(slices/2)*sliceStride] = Re[slices/2][0][0],
     * a[(slices/2)*sliceStride + 1] = Re[slices/2][0][columns/2],
     * a[(slices/2)*sliceStride + (rows/2)*rowStride] = Re[slices/2][rows/2][0],
     * a[(slices/2)*sliceStride + (rows/2)*rowStride + 1] =
     * Re[slices/2][rows/2][columns/2]
     * </pre>
     *  
     * This method computes only half of the elements of the real transform. The
     * other half satisfies the symmetry condition. If you want the full real
     * inverse transform, use <code>realInverseFull</code>.
     *  
     * @param a     data to transform
     *  
     * @param scale if true then scaling is performed
     */
    public void realInverse(double[] a, boolean scale)
    {
        if (isPowerOfTwo == false) {
            throw new IllegalArgumentException("slices, rows and columns must be power of two numbers");
        } else {
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && useThreads) {
                rdft3d_sub(-1, a);
                cdft3db_subth(1, a, scale);
                xdft3da_subth1(1, 1, a, scale);
            } else {
                rdft3d_sub(-1, a);
                cdft3db_sub(1, a, scale);
                xdft3da_sub1(1, 1, a, scale);
            }
        }
    }

    /**
     * Computes 3D inverse DFT of real data leaving the result in <code>a</code>
     * . This method only works when the sizes of all three dimensions are
     * power-of-two numbers. The data is stored in a 1D array addressed in
     * slice-major, then row-major, then column-major, in order of significance,
     * i.e. element (i,j,k) of 3-d array x[slices][rows][2*columns] is stored in
     * a[i*sliceStride + j*rowStride + k], where sliceStride = rows * 2 *
     * columns and rowStride = 2 * columns. The physical layout of the input
     * data has to be as follows:
     *  
     * <pre>
     * a[k1*sliceStride + k2*rowStride + 2*k3] = Re[k1][k2][k3] =
     * Re[(slices-k1)%slices][(rows-k2)%rows][columns-k3], a[k1*sliceStride +
     * k2*rowStride + 2*k3+1] = Im[k1][k2][k3] =
     * -Im[(slices-k1)%slices][(rows-k2)%rows][columns-k3], 0&lt;=k1&lt;slices,
     * 0&lt;=k2&lt;rows, 0&lt;k3&lt;columns/2, a[k1*sliceStride + k2*rowStride]
     * = Re[k1][k2][0] = Re[(slices-k1)%slices][rows-k2][0], a[k1*sliceStride +
     * k2*rowStride + 1] = Im[k1][k2][0] = -Im[(slices-k1)%slices][rows-k2][0],
     * a[k1*sliceStride + (rows-k2)*rowStride + 1] =
     * Re[(slices-k1)%slices][k2][columns/2] = Re[k1][rows-k2][columns/2],
     * a[k1*sliceStride + (rows-k2)*rowStride] =
     * -Im[(slices-k1)%slices][k2][columns/2] = Im[k1][rows-k2][columns/2],
     * 0&lt;=k1&lt;slices, 0&lt;k2&lt;rows/2, a[k1*sliceStride] = Re[k1][0][0] =
     * Re[slices-k1][0][0], a[k1*sliceStride + 1] = Im[k1][0][0] =
     * -Im[slices-k1][0][0], a[k1*sliceStride + (rows/2)*rowStride] =
     * Re[k1][rows/2][0] = Re[slices-k1][rows/2][0], a[k1*sliceStride +
     * (rows/2)*rowStride + 1] = Im[k1][rows/2][0] = -Im[slices-k1][rows/2][0],
     * a[(slices-k1)*sliceStride + 1] = Re[k1][0][columns/2] =
     * Re[slices-k1][0][columns/2], a[(slices-k1)*sliceStride] =
     * -Im[k1][0][columns/2] = Im[slices-k1][0][columns/2],
     * a[(slices-k1)*sliceStride + (rows/2)*rowStride + 1] =
     * Re[k1][rows/2][columns/2] = Re[slices-k1][rows/2][columns/2],
     * a[(slices-k1)*sliceStride + (rows/2) * rowStride] =
     * -Im[k1][rows/2][columns/2] = Im[slices-k1][rows/2][columns/2],
     * 0&lt;k1&lt;slices/2, a[0] = Re[0][0][0], a[1] = Re[0][0][columns/2],
     * a[(rows/2)*rowStride] = Re[0][rows/2][0], a[(rows/2)*rowStride + 1] =
     * Re[0][rows/2][columns/2], a[(slices/2)*sliceStride] = Re[slices/2][0][0],
     * a[(slices/2)*sliceStride + 1] = Re[slices/2][0][columns/2],
     * a[(slices/2)*sliceStride + (rows/2)*rowStride] = Re[slices/2][rows/2][0],
     * a[(slices/2)*sliceStride + (rows/2)*rowStride + 1] =
     * Re[slices/2][rows/2][columns/2]
     * </pre>
     *  
     * This method computes only half of the elements of the real transform. The
     * other half satisfies the symmetry condition. If you want the full real
     * inverse transform, use <code>realInverseFull</code>.
     *  
     * @param a     data to transform
     *  
     * @param scale if true then scaling is performed
     */
    public void realInverse(DoubleLargeArray a, boolean scale)
    {
        if (isPowerOfTwo == false) {
            throw new IllegalArgumentException("slices, rows and columns must be power of two numbers");
        } else {
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && useThreads) {
                rdft3d_sub(-1, a);
                cdft3db_subth(1, a, scale);
                xdft3da_subth1(1, 1, a, scale);
            } else {
                rdft3d_sub(-1, a);
                cdft3db_sub(1, a, scale);
                xdft3da_sub1(1, 1, a, scale);
            }
        }
    }

    /**
     * Computes 3D inverse DFT of real data leaving the result in <code>a</code>
     * . This method only works when the sizes of all three dimensions are
     * power-of-two numbers. The data is stored in a 3D array. The physical
     * layout of the input data has to be as follows:
     *  
     * <pre>
     * a[k1][k2][2*k3] = Re[k1][k2][k3] =
     * Re[(slices-k1)%slices][(rows-k2)%rows][columns-k3], a[k1][k2][2*k3+1] =
     * Im[k1][k2][k3] = -Im[(slices-k1)%slices][(rows-k2)%rows][columns-k3],
     * 0&lt;=k1&lt;slices, 0&lt;=k2&lt;rows, 0&lt;k3&lt;columns/2, a[k1][k2][0]
     * = Re[k1][k2][0] = Re[(slices-k1)%slices][rows-k2][0], a[k1][k2][1] =
     * Im[k1][k2][0] = -Im[(slices-k1)%slices][rows-k2][0], a[k1][rows-k2][1] =
     * Re[(slices-k1)%slices][k2][columns/2] = Re[k1][rows-k2][columns/2],
     * a[k1][rows-k2][0] = -Im[(slices-k1)%slices][k2][columns/2] =
     * Im[k1][rows-k2][columns/2], 0&lt;=k1&lt;slices, 0&lt;k2&lt;rows/2,
     * a[k1][0][0] = Re[k1][0][0] = Re[slices-k1][0][0], a[k1][0][1] =
     * Im[k1][0][0] = -Im[slices-k1][0][0], a[k1][rows/2][0] = Re[k1][rows/2][0]
     * = Re[slices-k1][rows/2][0], a[k1][rows/2][1] = Im[k1][rows/2][0] =
     * -Im[slices-k1][rows/2][0], a[slices-k1][0][1] = Re[k1][0][columns/2] =
     * Re[slices-k1][0][columns/2], a[slices-k1][0][0] = -Im[k1][0][columns/2] =
     * Im[slices-k1][0][columns/2], a[slices-k1][rows/2][1] =
     * Re[k1][rows/2][columns/2] = Re[slices-k1][rows/2][columns/2],
     * a[slices-k1][rows/2][0] = -Im[k1][rows/2][columns/2] =
     * Im[slices-k1][rows/2][columns/2], 0&lt;k1&lt;slices/2, a[0][0][0] =
     * Re[0][0][0], a[0][0][1] = Re[0][0][columns/2], a[0][rows/2][0] =
     * Re[0][rows/2][0], a[0][rows/2][1] = Re[0][rows/2][columns/2],
     * a[slices/2][0][0] = Re[slices/2][0][0], a[slices/2][0][1] =
     * Re[slices/2][0][columns/2], a[slices/2][rows/2][0] =
     * Re[slices/2][rows/2][0], a[slices/2][rows/2][1] =
     * Re[slices/2][rows/2][columns/2]
     * </pre>
     *  
     * This method computes only half of the elements of the real transform. The
     * other half satisfies the symmetry condition. If you want the full real
     * inverse transform, use <code>realInverseFull</code>.
     *  
     * @param a     data to transform
     *  
     * @param scale if true then scaling is performed
     */
    public void realInverse(double[][][] a, boolean scale)
    {
        if (isPowerOfTwo == false) {
            throw new IllegalArgumentException("slices, rows and columns must be power of two numbers");
        } else {
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && useThreads) {
                rdft3d_sub(-1, a);
                cdft3db_subth(1, a, scale);
                xdft3da_subth1(1, 1, a, scale);
            } else {
                rdft3d_sub(-1, a);
                cdft3db_sub(1, a, scale);
                xdft3da_sub1(1, 1, a, scale);
            }
        }
    }

    /**
     * Computes 3D inverse DFT of real data leaving the result in <code>a</code>
     * . This method computes full real inverse transform, i.e. you will get the
     * same result as from <code>complexInverse</code> called with all imaginary
     * part equal 0. Because the result is stored in <code>a</code>, the input
     * array must be of size slices*rows*2*columns, with only the first
     * slices*rows*columns elements filled with real data.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void realInverseFull(double[] a, boolean scale)
    {
        if (isPowerOfTwo) {
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && useThreads) {
                xdft3da_subth2(1, 1, a, scale);
                cdft3db_subth(1, a, scale);
                rdft3d_sub(1, a);
            } else {
                xdft3da_sub2(1, 1, a, scale);
                cdft3db_sub(1, a, scale);
                rdft3d_sub(1, a);
            }
            fillSymmetric(a);
        } else {
            mixedRadixRealInverseFull(a, scale);
        }
    }

    /**
     * Computes 3D inverse DFT of real data leaving the result in <code>a</code>
     * . This method computes full real inverse transform, i.e. you will get the
     * same result as from <code>complexInverse</code> called with all imaginary
     * part equal 0. Because the result is stored in <code>a</code>, the input
     * array must be of size slices*rows*2*columns, with only the first
     * slices*rows*columns elements filled with real data.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void realInverseFull(DoubleLargeArray a, boolean scale)
    {
        if (isPowerOfTwo) {
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && useThreads) {
                xdft3da_subth2(1, 1, a, scale);
                cdft3db_subth(1, a, scale);
                rdft3d_sub(1, a);
            } else {
                xdft3da_sub2(1, 1, a, scale);
                cdft3db_sub(1, a, scale);
                rdft3d_sub(1, a);
            }
            fillSymmetric(a);
        } else {
            mixedRadixRealInverseFull(a, scale);
        }
    }

    /**
     * Computes 3D inverse DFT of real data leaving the result in <code>a</code>
     * . This method computes full real inverse transform, i.e. you will get the
     * same result as from <code>complexInverse</code> called with all imaginary
     * part equal 0. Because the result is stored in <code>a</code>, the input
     * array must be of size slices by rows by 2*columns, with only the first
     * slices by rows by columns elements filled with real data.
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     */
    public void realInverseFull(double[][][] a, boolean scale)
    {
        if (isPowerOfTwo) {
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && useThreads) {
                xdft3da_subth2(1, 1, a, scale);
                cdft3db_subth(1, a, scale);
                rdft3d_sub(1, a);
            } else {
                xdft3da_sub2(1, 1, a, scale);
                cdft3db_sub(1, a, scale);
                rdft3d_sub(1, a);
            }
            fillSymmetric(a);
        } else {
            mixedRadixRealInverseFull(a, scale);
        }
    }

    /* -------- child routines -------- */
    private void mixedRadixRealForwardFull(final double[][][] a)
    {
        double[] temp = new double[2 * rows];
        int ldimn2 = rows / 2 + 1;
        final int newn3 = 2 * columns;
        final int n2d2;
        if (rows % 2 == 0) {
            n2d2 = rows / 2;
        } else {
            n2d2 = (rows + 1) / 2;
        }

        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && useThreads && (slices >= nthreads) && (columns >= nthreads) && (ldimn2 >= nthreads)) {
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
                                fftColumns.realForwardFull(a[s][r]);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;

                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        double[] temp = new double[2 * rows];

                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int c = 0; c < columns; c++) {
                                int idx2 = 2 * c;
                                for (int r = 0; r < rows; r++) {
                                    int idx4 = 2 * r;
                                    temp[idx4] = a[s][r][idx2];
                                    temp[idx4 + 1] = a[s][r][idx2 + 1];
                                }
                                fftRows.complexForward(temp);
                                for (int r = 0; r < rows; r++) {
                                    int idx4 = 2 * r;
                                    a[s][r][idx2] = temp[idx4];
                                    a[s][r][idx2 + 1] = temp[idx4 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            p = ldimn2 / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstRow = l * p;
                final int lastRow = (l == (nthreads - 1)) ? ldimn2 : firstRow + p;

                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        double[] temp = new double[2 * slices];

                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                int idx1 = 2 * c;
                                for (int s = 0; s < slices; s++) {
                                    int idx2 = 2 * s;
                                    temp[idx2] = a[s][r][idx1];
                                    temp[idx2 + 1] = a[s][r][idx1 + 1];
                                }
                                fftSlices.complexForward(temp);
                                for (int s = 0; s < slices; s++) {
                                    int idx2 = 2 * s;
                                    a[s][r][idx1] = temp[idx2];
                                    a[s][r][idx1 + 1] = temp[idx2 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }
            p = slices / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;

                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {

                        for (int s = firstSlice; s < lastSlice; s++) {
                            int idx2 = (slices - s) % slices;
                            for (int r = 1; r < n2d2; r++) {
                                int idx4 = rows - r;
                                for (int c = 0; c < columns; c++) {
                                    int idx1 = 2 * c;
                                    int idx3 = newn3 - idx1;
                                    a[idx2][idx4][idx3 % newn3] = a[s][r][idx1];
                                    a[idx2][idx4][(idx3 + 1) % newn3] = -a[s][r][idx1 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {

            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    fftColumns.realForwardFull(a[s][r]);
                }
            }

            for (int s = 0; s < slices; s++) {
                for (int c = 0; c < columns; c++) {
                    int idx2 = 2 * c;
                    for (int r = 0; r < rows; r++) {
                        int idx4 = 2 * r;
                        temp[idx4] = a[s][r][idx2];
                        temp[idx4 + 1] = a[s][r][idx2 + 1];
                    }
                    fftRows.complexForward(temp);
                    for (int r = 0; r < rows; r++) {
                        int idx4 = 2 * r;
                        a[s][r][idx2] = temp[idx4];
                        a[s][r][idx2 + 1] = temp[idx4 + 1];
                    }
                }
            }

            temp = new double[2 * slices];

            for (int r = 0; r < ldimn2; r++) {
                for (int c = 0; c < columns; c++) {
                    int idx1 = 2 * c;
                    for (int s = 0; s < slices; s++) {
                        int idx2 = 2 * s;
                        temp[idx2] = a[s][r][idx1];
                        temp[idx2 + 1] = a[s][r][idx1 + 1];
                    }
                    fftSlices.complexForward(temp);
                    for (int s = 0; s < slices; s++) {
                        int idx2 = 2 * s;
                        a[s][r][idx1] = temp[idx2];
                        a[s][r][idx1 + 1] = temp[idx2 + 1];
                    }
                }
            }

            for (int s = 0; s < slices; s++) {
                int idx2 = (slices - s) % slices;
                for (int r = 1; r < n2d2; r++) {
                    int idx4 = rows - r;
                    for (int c = 0; c < columns; c++) {
                        int idx1 = 2 * c;
                        int idx3 = newn3 - idx1;
                        a[idx2][idx4][idx3 % newn3] = a[s][r][idx1];
                        a[idx2][idx4][(idx3 + 1) % newn3] = -a[s][r][idx1 + 1];
                    }
                }
            }

        }
    }

    private void mixedRadixRealInverseFull(final double[][][] a, final boolean scale)
    {
        double[] temp = new double[2 * rows];
        int ldimn2 = rows / 2 + 1;
        final int newn3 = 2 * columns;
        final int n2d2;
        if (rows % 2 == 0) {
            n2d2 = rows / 2;
        } else {
            n2d2 = (rows + 1) / 2;
        }

        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && useThreads && (slices >= nthreads) && (columns >= nthreads) && (ldimn2 >= nthreads)) {
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
                                fftColumns.realInverseFull(a[s][r], scale);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;

                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        double[] temp = new double[2 * rows];

                        for (int s = firstSlice; s < lastSlice; s++) {
                            for (int c = 0; c < columns; c++) {
                                int idx2 = 2 * c;
                                for (int r = 0; r < rows; r++) {
                                    int idx4 = 2 * r;
                                    temp[idx4] = a[s][r][idx2];
                                    temp[idx4 + 1] = a[s][r][idx2 + 1];
                                }
                                fftRows.complexInverse(temp, scale);
                                for (int r = 0; r < rows; r++) {
                                    int idx4 = 2 * r;
                                    a[s][r][idx2] = temp[idx4];
                                    a[s][r][idx2 + 1] = temp[idx4 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            p = ldimn2 / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstRow = l * p;
                final int lastRow = (l == (nthreads - 1)) ? ldimn2 : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        double[] temp = new double[2 * slices];

                        for (int r = firstRow; r < lastRow; r++) {
                            for (int c = 0; c < columns; c++) {
                                int idx1 = 2 * c;
                                for (int s = 0; s < slices; s++) {
                                    int idx2 = 2 * s;
                                    temp[idx2] = a[s][r][idx1];
                                    temp[idx2 + 1] = a[s][r][idx1 + 1];
                                }
                                fftSlices.complexInverse(temp, scale);
                                for (int s = 0; s < slices; s++) {
                                    int idx2 = 2 * s;
                                    a[s][r][idx1] = temp[idx2];
                                    a[s][r][idx1 + 1] = temp[idx2 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }
            p = slices / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;

                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {

                        for (int s = firstSlice; s < lastSlice; s++) {
                            int idx2 = (slices - s) % slices;
                            for (int r = 1; r < n2d2; r++) {
                                int idx4 = rows - r;
                                for (int c = 0; c < columns; c++) {
                                    int idx1 = 2 * c;
                                    int idx3 = newn3 - idx1;
                                    a[idx2][idx4][idx3 % newn3] = a[s][r][idx1];
                                    a[idx2][idx4][(idx3 + 1) % newn3] = -a[s][r][idx1 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {

            for (int s = 0; s < slices; s++) {
                for (int r = 0; r < rows; r++) {
                    fftColumns.realInverseFull(a[s][r], scale);
                }
            }

            for (int s = 0; s < slices; s++) {
                for (int c = 0; c < columns; c++) {
                    int idx2 = 2 * c;
                    for (int r = 0; r < rows; r++) {
                        int idx4 = 2 * r;
                        temp[idx4] = a[s][r][idx2];
                        temp[idx4 + 1] = a[s][r][idx2 + 1];
                    }
                    fftRows.complexInverse(temp, scale);
                    for (int r = 0; r < rows; r++) {
                        int idx4 = 2 * r;
                        a[s][r][idx2] = temp[idx4];
                        a[s][r][idx2 + 1] = temp[idx4 + 1];
                    }
                }
            }

            temp = new double[2 * slices];

            for (int r = 0; r < ldimn2; r++) {
                for (int c = 0; c < columns; c++) {
                    int idx1 = 2 * c;
                    for (int s = 0; s < slices; s++) {
                        int idx2 = 2 * s;
                        temp[idx2] = a[s][r][idx1];
                        temp[idx2 + 1] = a[s][r][idx1 + 1];
                    }
                    fftSlices.complexInverse(temp, scale);
                    for (int s = 0; s < slices; s++) {
                        int idx2 = 2 * s;
                        a[s][r][idx1] = temp[idx2];
                        a[s][r][idx1 + 1] = temp[idx2 + 1];
                    }
                }
            }

            for (int s = 0; s < slices; s++) {
                int idx2 = (slices - s) % slices;
                for (int r = 1; r < n2d2; r++) {
                    int idx4 = rows - r;
                    for (int c = 0; c < columns; c++) {
                        int idx1 = 2 * c;
                        int idx3 = newn3 - idx1;
                        a[idx2][idx4][idx3 % newn3] = a[s][r][idx1];
                        a[idx2][idx4][(idx3 + 1) % newn3] = -a[s][r][idx1 + 1];
                    }
                }
            }

        }
    }

    private void mixedRadixRealForwardFull(final double[] a)
    {
        final int twon3 = 2 * columns;
        double[] temp = new double[twon3];
        int ldimn2 = rows / 2 + 1;
        final int n2d2;
        if (rows % 2 == 0) {
            n2d2 = rows / 2;
        } else {
            n2d2 = (rows + 1) / 2;
        }

        final int twoSliceStride = 2 * sliceStride;
        final int twoRowStride = 2 * rowStride;
        int n1d2 = slices / 2;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && useThreads && (n1d2 >= nthreads) && (columns >= nthreads) && (ldimn2 >= nthreads)) {
            Future<?>[] futures = new Future[nthreads];
            int p = n1d2 / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = slices - 1 - l * p;
                final int lastSlice = (l == (nthreads - 1)) ? n1d2 + 1 : firstSlice - p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        double[] temp = new double[twon3];
                        for (int s = firstSlice; s >= lastSlice; s--) {
                            int idx1 = s * sliceStride;
                            int idx2 = s * twoSliceStride;
                            for (int r = rows - 1; r >= 0; r--) {
                                System.arraycopy(a, idx1 + r * rowStride, temp, 0, columns);
                                fftColumns.realForwardFull(temp);
                                System.arraycopy(temp, 0, a, idx2 + r * twoRowStride, twon3);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            final double[][][] temp2 = new double[n1d2 + 1][rows][twon3];

            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? n1d2 + 1 : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (int s = firstSlice; s < lastSlice; s++) {
                            int idx1 = s * sliceStride;
                            for (int r = 0; r < rows; r++) {
                                System.arraycopy(a, idx1 + r * rowStride, temp2[s][r], 0, columns);
                                fftColumns.realForwardFull(temp2[s][r]);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? n1d2 + 1 : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (int s = firstSlice; s < lastSlice; s++) {
                            int idx1 = s * twoSliceStride;
                            for (int r = 0; r < rows; r++) {
                                System.arraycopy(temp2[s][r], 0, a, idx1 + r * twoRowStride, twon3);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            p = slices / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        double[] temp = new double[2 * rows];

                        for (int s = firstSlice; s < lastSlice; s++) {
                            int idx1 = s * twoSliceStride;
                            for (int c = 0; c < columns; c++) {
                                int idx2 = 2 * c;
                                for (int r = 0; r < rows; r++) {
                                    int idx3 = idx1 + r * twoRowStride + idx2;
                                    int idx4 = 2 * r;
                                    temp[idx4] = a[idx3];
                                    temp[idx4 + 1] = a[idx3 + 1];
                                }
                                fftRows.complexForward(temp);
                                for (int r = 0; r < rows; r++) {
                                    int idx3 = idx1 + r * twoRowStride + idx2;
                                    int idx4 = 2 * r;
                                    a[idx3] = temp[idx4];
                                    a[idx3 + 1] = temp[idx4 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            p = ldimn2 / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstRow = l * p;
                final int lastRow = (l == (nthreads - 1)) ? ldimn2 : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        double[] temp = new double[2 * slices];

                        for (int r = firstRow; r < lastRow; r++) {
                            int idx3 = r * twoRowStride;
                            for (int c = 0; c < columns; c++) {
                                int idx1 = 2 * c;
                                for (int s = 0; s < slices; s++) {
                                    int idx2 = 2 * s;
                                    int idx4 = s * twoSliceStride + idx3 + idx1;
                                    temp[idx2] = a[idx4];
                                    temp[idx2 + 1] = a[idx4 + 1];
                                }
                                fftSlices.complexForward(temp);
                                for (int s = 0; s < slices; s++) {
                                    int idx2 = 2 * s;
                                    int idx4 = s * twoSliceStride + idx3 + idx1;
                                    a[idx4] = temp[idx2];
                                    a[idx4 + 1] = temp[idx2 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }
            p = slices / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {

                        for (int s = firstSlice; s < lastSlice; s++) {
                            int idx2 = (slices - s) % slices;
                            int idx5 = idx2 * twoSliceStride;
                            int idx6 = s * twoSliceStride;
                            for (int r = 1; r < n2d2; r++) {
                                int idx4 = rows - r;
                                int idx7 = idx4 * twoRowStride;
                                int idx8 = r * twoRowStride;
                                int idx9 = idx5 + idx7;
                                for (int c = 0; c < columns; c++) {
                                    int idx1 = 2 * c;
                                    int idx3 = twon3 - idx1;
                                    int idx10 = idx6 + idx8 + idx1;
                                    a[idx9 + idx3 % twon3] = a[idx10];
                                    a[idx9 + (idx3 + 1) % twon3] = -a[idx10 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {

            for (int s = slices - 1; s >= 0; s--) {
                int idx1 = s * sliceStride;
                int idx2 = s * twoSliceStride;
                for (int r = rows - 1; r >= 0; r--) {
                    System.arraycopy(a, idx1 + r * rowStride, temp, 0, columns);
                    fftColumns.realForwardFull(temp);
                    System.arraycopy(temp, 0, a, idx2 + r * twoRowStride, twon3);
                }
            }

            temp = new double[2 * rows];

            for (int s = 0; s < slices; s++) {
                int idx1 = s * twoSliceStride;
                for (int c = 0; c < columns; c++) {
                    int idx2 = 2 * c;
                    for (int r = 0; r < rows; r++) {
                        int idx4 = 2 * r;
                        int idx3 = idx1 + r * twoRowStride + idx2;
                        temp[idx4] = a[idx3];
                        temp[idx4 + 1] = a[idx3 + 1];
                    }
                    fftRows.complexForward(temp);
                    for (int r = 0; r < rows; r++) {
                        int idx4 = 2 * r;
                        int idx3 = idx1 + r * twoRowStride + idx2;
                        a[idx3] = temp[idx4];
                        a[idx3 + 1] = temp[idx4 + 1];
                    }
                }
            }

            temp = new double[2 * slices];

            for (int r = 0; r < ldimn2; r++) {
                int idx3 = r * twoRowStride;
                for (int c = 0; c < columns; c++) {
                    int idx1 = 2 * c;
                    for (int s = 0; s < slices; s++) {
                        int idx2 = 2 * s;
                        int idx4 = s * twoSliceStride + idx3 + idx1;
                        temp[idx2] = a[idx4];
                        temp[idx2 + 1] = a[idx4 + 1];
                    }
                    fftSlices.complexForward(temp);
                    for (int s = 0; s < slices; s++) {
                        int idx2 = 2 * s;
                        int idx4 = s * twoSliceStride + idx3 + idx1;
                        a[idx4] = temp[idx2];
                        a[idx4 + 1] = temp[idx2 + 1];
                    }
                }
            }

            for (int s = 0; s < slices; s++) {
                int idx2 = (slices - s) % slices;
                int idx5 = idx2 * twoSliceStride;
                int idx6 = s * twoSliceStride;
                for (int r = 1; r < n2d2; r++) {
                    int idx4 = rows - r;
                    int idx7 = idx4 * twoRowStride;
                    int idx8 = r * twoRowStride;
                    int idx9 = idx5 + idx7;
                    for (int c = 0; c < columns; c++) {
                        int idx1 = 2 * c;
                        int idx3 = twon3 - idx1;
                        int idx10 = idx6 + idx8 + idx1;
                        a[idx9 + idx3 % twon3] = a[idx10];
                        a[idx9 + (idx3 + 1) % twon3] = -a[idx10 + 1];
                    }
                }
            }

        }
    }

    private void mixedRadixRealForwardFull(final DoubleLargeArray a)
    {
        final long twon3 = 2 * columnsl;
        DoubleLargeArray temp = new DoubleLargeArray(twon3);
        long ldimn2 = rowsl / 2 + 1;
        final long n2d2;
        if (rowsl % 2 == 0) {
            n2d2 = rowsl / 2;
        } else {
            n2d2 = (rowsl + 1) / 2;
        }

        final long twoSliceStride = 2 * sliceStridel;
        final long twoRowStride = 2 * rowStridel;
        long n1d2 = slicesl / 2;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && useThreads && (n1d2 >= nthreads) && (columnsl >= nthreads) && (ldimn2 >= nthreads)) {
            Future<?>[] futures = new Future[nthreads];
            long p = n1d2 / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final long firstSlice = slicesl - 1 - l * p;
                final long lastSlice = (l == (nthreads - 1)) ? n1d2 + 1 : firstSlice - p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        DoubleLargeArray temp = new DoubleLargeArray(twon3);
                        for (long s = firstSlice; s >= lastSlice; s--) {
                            long idx1 = s * sliceStridel;
                            long idx2 = s * twoSliceStride;
                            for (long r = rowsl - 1; r >= 0; r--) {
                                LargeArrayUtils.arraycopy(a, idx1 + r * rowStridel, temp, 0, columnsl);
                                fftColumns.realForwardFull(temp);
                                LargeArrayUtils.arraycopy(temp, 0, a, idx2 + r * twoRowStride, twon3);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            final DoubleLargeArray temp2 = new DoubleLargeArray((n1d2 + 1) * rowsl * twon3);

            for (int l = 0; l < nthreads; l++) {
                final long firstSlice = l * p;
                final long lastSlice = (l == (nthreads - 1)) ? n1d2 + 1 : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (long s = firstSlice; s < lastSlice; s++) {
                            long idx1 = s * sliceStridel;
                            for (long r = 0; r < rowsl; r++) {
                                LargeArrayUtils.arraycopy(a, idx1 + r * rowStridel, temp2, s * rowsl * twon3 + r * twon3, columnsl);
                                fftColumns.realForwardFull(temp2, s * rowsl * twon3 + r * twon3);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int l = 0; l < nthreads; l++) {
                final long firstSlice = l * p;
                final long lastSlice = (l == (nthreads - 1)) ? n1d2 + 1 : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (long s = firstSlice; s < lastSlice; s++) {
                            long idx1 = s * twoSliceStride;
                            for (long r = 0; r < rowsl; r++) {
                                LargeArrayUtils.arraycopy(temp2, s * rowsl * twon3 + r * twon3, a, idx1 + r * twoRowStride, twon3);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            p = slicesl / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final long firstSlice = l * p;
                final long lastSlice = (l == (nthreads - 1)) ? slicesl : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        DoubleLargeArray temp = new DoubleLargeArray(2 * rowsl, false);

                        for (long s = firstSlice; s < lastSlice; s++) {
                            long idx1 = s * twoSliceStride;
                            for (long c = 0; c < columnsl; c++) {
                                long idx2 = 2 * c;
                                for (long r = 0; r < rowsl; r++) {
                                    long idx3 = idx1 + r * twoRowStride + idx2;
                                    long idx4 = 2 * r;
                                    temp.setDouble(idx4, a.getDouble(idx3));
                                    temp.setDouble(idx4 + 1, a.getDouble(idx3 + 1));
                                }
                                fftRows.complexForward(temp);
                                for (long r = 0; r < rowsl; r++) {
                                    long idx3 = idx1 + r * twoRowStride + idx2;
                                    long idx4 = 2 * r;
                                    a.setDouble(idx3, temp.getDouble(idx4));
                                    a.setDouble(idx3 + 1, temp.getDouble(idx4 + 1));
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            p = ldimn2 / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final long firstRow = l * p;
                final long lastRow = (l == (nthreads - 1)) ? ldimn2 : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        DoubleLargeArray temp = new DoubleLargeArray(2 * slicesl, false);

                        for (long r = firstRow; r < lastRow; r++) {
                            long idx3 = r * twoRowStride;
                            for (long c = 0; c < columnsl; c++) {
                                long idx1 = 2 * c;
                                for (long s = 0; s < slicesl; s++) {
                                    long idx2 = 2 * s;
                                    long idx4 = s * twoSliceStride + idx3 + idx1;
                                    temp.setDouble(idx2, a.getDouble(idx4));
                                    temp.setDouble(idx2 + 1, a.getDouble(idx4 + 1));
                                }
                                fftSlices.complexForward(temp);
                                for (long s = 0; s < slicesl; s++) {
                                    long idx2 = 2 * s;
                                    long idx4 = s * twoSliceStride + idx3 + idx1;
                                    a.setDouble(idx4, temp.getDouble(idx2));
                                    a.setDouble(idx4 + 1, temp.getDouble(idx2 + 1));
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }
            p = slicesl / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final long firstSlice = l * p;
                final long lastSlice = (l == (nthreads - 1)) ? slicesl : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {

                        for (long s = firstSlice; s < lastSlice; s++) {
                            long idx2 = (slicesl - s) % slicesl;
                            long idx5 = idx2 * twoSliceStride;
                            long idx6 = s * twoSliceStride;
                            for (long r = 1; r < n2d2; r++) {
                                long idx4 = rowsl - r;
                                long idx7 = idx4 * twoRowStride;
                                long idx8 = r * twoRowStride;
                                long idx9 = idx5 + idx7;
                                for (long c = 0; c < columnsl; c++) {
                                    long idx1 = 2 * c;
                                    long idx3 = twon3 - idx1;
                                    long idx10 = idx6 + idx8 + idx1;
                                    a.setDouble(idx9 + idx3 % twon3, a.getDouble(idx10));
                                    a.setDouble(idx9 + (idx3 + 1) % twon3, -a.getDouble(idx10 + 1));
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {

            for (long s = slicesl - 1; s >= 0; s--) {
                long idx1 = s * sliceStridel;
                long idx2 = s * twoSliceStride;
                for (long r = rowsl - 1; r >= 0; r--) {
                    LargeArrayUtils.arraycopy(a, idx1 + r * rowStridel, temp, 0, columnsl);
                    fftColumns.realForwardFull(temp);
                    LargeArrayUtils.arraycopy(temp, 0, a, idx2 + r * twoRowStride, twon3);
                }
            }

            temp = new DoubleLargeArray(2 * rowsl, false);

            for (long s = 0; s < slicesl; s++) {
                long idx1 = s * twoSliceStride;
                for (long c = 0; c < columnsl; c++) {
                    long idx2 = 2 * c;
                    for (long r = 0; r < rowsl; r++) {
                        long idx4 = 2 * r;
                        long idx3 = idx1 + r * twoRowStride + idx2;
                        temp.setDouble(idx4, a.getDouble(idx3));
                        temp.setDouble(idx4 + 1, a.getDouble(idx3 + 1));
                    }
                    fftRows.complexForward(temp);
                    for (long r = 0; r < rowsl; r++) {
                        long idx4 = 2 * r;
                        long idx3 = idx1 + r * twoRowStride + idx2;
                        a.setDouble(idx3, temp.getDouble(idx4));
                        a.setDouble(idx3 + 1, temp.getDouble(idx4 + 1));
                    }
                }
            }

            temp = new DoubleLargeArray(2 * slicesl, false);

            for (long r = 0; r < ldimn2; r++) {
                long idx3 = r * twoRowStride;
                for (long c = 0; c < columnsl; c++) {
                    long idx1 = 2 * c;
                    for (long s = 0; s < slicesl; s++) {
                        long idx2 = 2 * s;
                        long idx4 = s * twoSliceStride + idx3 + idx1;
                        temp.setDouble(idx2, a.getDouble(idx4));
                        temp.setDouble(idx2 + 1, a.getDouble(idx4 + 1));
                    }
                    fftSlices.complexForward(temp);
                    for (long s = 0; s < slicesl; s++) {
                        long idx2 = 2 * s;
                        long idx4 = s * twoSliceStride + idx3 + idx1;
                        a.setDouble(idx4, temp.getDouble(idx2));
                        a.setDouble(idx4 + 1, temp.getDouble(idx2 + 1));
                    }
                }
            }

            for (long s = 0; s < slicesl; s++) {
                long idx2 = (slicesl - s) % slicesl;
                long idx5 = idx2 * twoSliceStride;
                long idx6 = s * twoSliceStride;
                for (long r = 1; r < n2d2; r++) {
                    long idx4 = rowsl - r;
                    long idx7 = idx4 * twoRowStride;
                    long idx8 = r * twoRowStride;
                    long idx9 = idx5 + idx7;
                    for (long c = 0; c < columnsl; c++) {
                        long idx1 = 2 * c;
                        long idx3 = twon3 - idx1;
                        long idx10 = idx6 + idx8 + idx1;
                        a.setDouble(idx9 + idx3 % twon3, a.getDouble(idx10));
                        a.setDouble(idx9 + (idx3 + 1) % twon3, -a.getDouble(idx10 + 1));
                    }
                }
            }

        }
    }

    private void mixedRadixRealInverseFull(final double[] a, final boolean scale)
    {
        final int twon3 = 2 * columns;
        double[] temp = new double[twon3];
        int ldimn2 = rows / 2 + 1;
        final int n2d2;
        if (rows % 2 == 0) {
            n2d2 = rows / 2;
        } else {
            n2d2 = (rows + 1) / 2;
        }

        final int twoSliceStride = 2 * sliceStride;
        final int twoRowStride = 2 * rowStride;
        int n1d2 = slices / 2;

        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && useThreads && (n1d2 >= nthreads) && (columns >= nthreads) && (ldimn2 >= nthreads)) {
            Future<?>[] futures = new Future[nthreads];
            int p = n1d2 / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = slices - 1 - l * p;
                final int lastSlice = (l == (nthreads - 1)) ? n1d2 + 1 : firstSlice - p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        double[] temp = new double[twon3];
                        for (int s = firstSlice; s >= lastSlice; s--) {
                            int idx1 = s * sliceStride;
                            int idx2 = s * twoSliceStride;
                            for (int r = rows - 1; r >= 0; r--) {
                                System.arraycopy(a, idx1 + r * rowStride, temp, 0, columns);
                                fftColumns.realInverseFull(temp, scale);
                                System.arraycopy(temp, 0, a, idx2 + r * twoRowStride, twon3);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            final double[][][] temp2 = new double[n1d2 + 1][rows][twon3];

            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? n1d2 + 1 : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (int s = firstSlice; s < lastSlice; s++) {
                            int idx1 = s * sliceStride;
                            for (int r = 0; r < rows; r++) {
                                System.arraycopy(a, idx1 + r * rowStride, temp2[s][r], 0, columns);
                                fftColumns.realInverseFull(temp2[s][r], scale);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? n1d2 + 1 : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (int s = firstSlice; s < lastSlice; s++) {
                            int idx1 = s * twoSliceStride;
                            for (int r = 0; r < rows; r++) {
                                System.arraycopy(temp2[s][r], 0, a, idx1 + r * twoRowStride, twon3);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            p = slices / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        double[] temp = new double[2 * rows];

                        for (int s = firstSlice; s < lastSlice; s++) {
                            int idx1 = s * twoSliceStride;
                            for (int c = 0; c < columns; c++) {
                                int idx2 = 2 * c;
                                for (int r = 0; r < rows; r++) {
                                    int idx3 = idx1 + r * twoRowStride + idx2;
                                    int idx4 = 2 * r;
                                    temp[idx4] = a[idx3];
                                    temp[idx4 + 1] = a[idx3 + 1];
                                }
                                fftRows.complexInverse(temp, scale);
                                for (int r = 0; r < rows; r++) {
                                    int idx3 = idx1 + r * twoRowStride + idx2;
                                    int idx4 = 2 * r;
                                    a[idx3] = temp[idx4];
                                    a[idx3 + 1] = temp[idx4 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            p = ldimn2 / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstRow = l * p;
                final int lastRow = (l == (nthreads - 1)) ? ldimn2 : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        double[] temp = new double[2 * slices];

                        for (int r = firstRow; r < lastRow; r++) {
                            int idx3 = r * twoRowStride;
                            for (int c = 0; c < columns; c++) {
                                int idx1 = 2 * c;
                                for (int s = 0; s < slices; s++) {
                                    int idx2 = 2 * s;
                                    int idx4 = s * twoSliceStride + idx3 + idx1;
                                    temp[idx2] = a[idx4];
                                    temp[idx2 + 1] = a[idx4 + 1];
                                }
                                fftSlices.complexInverse(temp, scale);
                                for (int s = 0; s < slices; s++) {
                                    int idx2 = 2 * s;
                                    int idx4 = s * twoSliceStride + idx3 + idx1;
                                    a[idx4] = temp[idx2];
                                    a[idx4 + 1] = temp[idx2 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            p = slices / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {

                        for (int s = firstSlice; s < lastSlice; s++) {
                            int idx2 = (slices - s) % slices;
                            int idx5 = idx2 * twoSliceStride;
                            int idx6 = s * twoSliceStride;
                            for (int r = 1; r < n2d2; r++) {
                                int idx4 = rows - r;
                                int idx7 = idx4 * twoRowStride;
                                int idx8 = r * twoRowStride;
                                int idx9 = idx5 + idx7;
                                for (int c = 0; c < columns; c++) {
                                    int idx1 = 2 * c;
                                    int idx3 = twon3 - idx1;
                                    int idx10 = idx6 + idx8 + idx1;
                                    a[idx9 + idx3 % twon3] = a[idx10];
                                    a[idx9 + (idx3 + 1) % twon3] = -a[idx10 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {

            for (int s = slices - 1; s >= 0; s--) {
                int idx1 = s * sliceStride;
                int idx2 = s * twoSliceStride;
                for (int r = rows - 1; r >= 0; r--) {
                    System.arraycopy(a, idx1 + r * rowStride, temp, 0, columns);
                    fftColumns.realInverseFull(temp, scale);
                    System.arraycopy(temp, 0, a, idx2 + r * twoRowStride, twon3);
                }
            }

            temp = new double[2 * rows];

            for (int s = 0; s < slices; s++) {
                int idx1 = s * twoSliceStride;
                for (int c = 0; c < columns; c++) {
                    int idx2 = 2 * c;
                    for (int r = 0; r < rows; r++) {
                        int idx4 = 2 * r;
                        int idx3 = idx1 + r * twoRowStride + idx2;
                        temp[idx4] = a[idx3];
                        temp[idx4 + 1] = a[idx3 + 1];
                    }
                    fftRows.complexInverse(temp, scale);
                    for (int r = 0; r < rows; r++) {
                        int idx4 = 2 * r;
                        int idx3 = idx1 + r * twoRowStride + idx2;
                        a[idx3] = temp[idx4];
                        a[idx3 + 1] = temp[idx4 + 1];
                    }
                }
            }

            temp = new double[2 * slices];

            for (int r = 0; r < ldimn2; r++) {
                int idx3 = r * twoRowStride;
                for (int c = 0; c < columns; c++) {
                    int idx1 = 2 * c;
                    for (int s = 0; s < slices; s++) {
                        int idx2 = 2 * s;
                        int idx4 = s * twoSliceStride + idx3 + idx1;
                        temp[idx2] = a[idx4];
                        temp[idx2 + 1] = a[idx4 + 1];
                    }
                    fftSlices.complexInverse(temp, scale);
                    for (int s = 0; s < slices; s++) {
                        int idx2 = 2 * s;
                        int idx4 = s * twoSliceStride + idx3 + idx1;
                        a[idx4] = temp[idx2];
                        a[idx4 + 1] = temp[idx2 + 1];
                    }
                }
            }

            for (int s = 0; s < slices; s++) {
                int idx2 = (slices - s) % slices;
                int idx5 = idx2 * twoSliceStride;
                int idx6 = s * twoSliceStride;
                for (int r = 1; r < n2d2; r++) {
                    int idx4 = rows - r;
                    int idx7 = idx4 * twoRowStride;
                    int idx8 = r * twoRowStride;
                    int idx9 = idx5 + idx7;
                    for (int c = 0; c < columns; c++) {
                        int idx1 = 2 * c;
                        int idx3 = twon3 - idx1;
                        int idx10 = idx6 + idx8 + idx1;
                        a[idx9 + idx3 % twon3] = a[idx10];
                        a[idx9 + (idx3 + 1) % twon3] = -a[idx10 + 1];
                    }
                }
            }

        }
    }

    private void mixedRadixRealInverseFull(final DoubleLargeArray a, final boolean scale)
    {
        final long twon3 = 2 * columnsl;
        DoubleLargeArray temp = new DoubleLargeArray(twon3);
        long ldimn2 = rowsl / 2 + 1;
        final long n2d2;
        if (rowsl % 2 == 0) {
            n2d2 = rowsl / 2;
        } else {
            n2d2 = (rowsl + 1) / 2;
        }

        final long twoSliceStride = 2 * sliceStridel;
        final long twoRowStride = 2 * rowStridel;
        long n1d2 = slicesl / 2;

        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && useThreads && (n1d2 >= nthreads) && (columnsl >= nthreads) && (ldimn2 >= nthreads)) {
            Future<?>[] futures = new Future[nthreads];
            long p = n1d2 / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final long firstSlice = slicesl - 1 - l * p;
                final long lastSlice = (l == (nthreads - 1)) ? n1d2 + 1 : firstSlice - p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        DoubleLargeArray temp = new DoubleLargeArray(twon3);
                        for (long s = firstSlice; s >= lastSlice; s--) {
                            long idx1 = s * sliceStridel;
                            long idx2 = s * twoSliceStride;
                            for (long r = rowsl - 1; r >= 0; r--) {
                                LargeArrayUtils.arraycopy(a, idx1 + r * rowStridel, temp, 0, columnsl);
                                fftColumns.realInverseFull(temp, scale);
                                LargeArrayUtils.arraycopy(temp, 0, a, idx2 + r * twoRowStride, twon3);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            final DoubleLargeArray temp2 = new DoubleLargeArray((n1d2 + 1) * rowsl * twon3);

            for (int l = 0; l < nthreads; l++) {
                final long firstSlice = l * p;
                final long lastSlice = (l == (nthreads - 1)) ? n1d2 + 1 : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (long s = firstSlice; s < lastSlice; s++) {
                            long idx1 = s * sliceStridel;
                            for (long r = 0; r < rowsl; r++) {
                                LargeArrayUtils.arraycopy(a, idx1 + r * rowStridel, temp2, s * rowsl * twon3 + r * twon3, columnsl);
                                fftColumns.realInverseFull(temp2, s * rowsl * twon3 + r * twon3, scale);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int l = 0; l < nthreads; l++) {
                final long firstSlice = l * p;
                final long lastSlice = (l == (nthreads - 1)) ? n1d2 + 1 : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (long s = firstSlice; s < lastSlice; s++) {
                            long idx1 = s * twoSliceStride;
                            for (long r = 0; r < rowsl; r++) {
                                LargeArrayUtils.arraycopy(temp2, s * rowsl * twon3 + r * twon3, a, idx1 + r * twoRowStride, twon3);
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            p = slicesl / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final long firstSlice = l * p;
                final long lastSlice = (l == (nthreads - 1)) ? slicesl : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        DoubleLargeArray temp = new DoubleLargeArray(2 * rowsl, false);

                        for (long s = firstSlice; s < lastSlice; s++) {
                            long idx1 = s * twoSliceStride;
                            for (long c = 0; c < columnsl; c++) {
                                long idx2 = 2 * c;
                                for (long r = 0; r < rowsl; r++) {
                                    long idx3 = idx1 + r * twoRowStride + idx2;
                                    long idx4 = 2 * r;
                                    temp.setDouble(idx4, a.getDouble(idx3));
                                    temp.setDouble(idx4 + 1, a.getDouble(idx3 + 1));
                                }
                                fftRows.complexInverse(temp, scale);
                                for (long r = 0; r < rowsl; r++) {
                                    long idx3 = idx1 + r * twoRowStride + idx2;
                                    long idx4 = 2 * r;
                                    a.setDouble(idx3, temp.getDouble(idx4));
                                    a.setDouble(idx3 + 1, temp.getDouble(idx4 + 1));
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            p = ldimn2 / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final long firstRow = l * p;
                final long lastRow = (l == (nthreads - 1)) ? ldimn2 : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        DoubleLargeArray temp = new DoubleLargeArray(2 * slicesl, false);

                        for (long r = firstRow; r < lastRow; r++) {
                            long idx3 = r * twoRowStride;
                            for (long c = 0; c < columnsl; c++) {
                                long idx1 = 2 * c;
                                for (long s = 0; s < slicesl; s++) {
                                    long idx2 = 2 * s;
                                    long idx4 = s * twoSliceStride + idx3 + idx1;
                                    temp.setDouble(idx2, a.getDouble(idx4));
                                    temp.setDouble(idx2 + 1, a.getDouble(idx4 + 1));
                                }
                                fftSlices.complexInverse(temp, scale);
                                for (long s = 0; s < slicesl; s++) {
                                    long idx2 = 2 * s;
                                    long idx4 = s * twoSliceStride + idx3 + idx1;
                                    a.setDouble(idx4, temp.getDouble(idx2));
                                    a.setDouble(idx4 + 1, temp.getDouble(idx2 + 1));
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            p = slicesl / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final long firstSlice = l * p;
                final long lastSlice = (l == (nthreads - 1)) ? slicesl : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {

                        for (long s = firstSlice; s < lastSlice; s++) {
                            long idx2 = (slicesl - s) % slicesl;
                            long idx5 = idx2 * twoSliceStride;
                            long idx6 = s * twoSliceStride;
                            for (long r = 1; r < n2d2; r++) {
                                long idx4 = rowsl - r;
                                long idx7 = idx4 * twoRowStride;
                                long idx8 = r * twoRowStride;
                                long idx9 = idx5 + idx7;
                                for (long c = 0; c < columnsl; c++) {
                                    long idx1 = 2 * c;
                                    long idx3 = twon3 - idx1;
                                    long idx10 = idx6 + idx8 + idx1;
                                    a.setDouble(idx9 + idx3 % twon3, a.getDouble(idx10));
                                    a.setDouble(idx9 + (idx3 + 1) % twon3, -a.getDouble(idx10 + 1));
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {

            for (long s = slicesl - 1; s >= 0; s--) {
                long idx1 = s * sliceStridel;
                long idx2 = s * twoSliceStride;
                for (long r = rowsl - 1; r >= 0; r--) {
                    LargeArrayUtils.arraycopy(a, idx1 + r * rowStridel, temp, 0, columnsl);
                    fftColumns.realInverseFull(temp, scale);
                    LargeArrayUtils.arraycopy(temp, 0, a, idx2 + r * twoRowStride, twon3);
                }
            }

            temp = new DoubleLargeArray(2 * rowsl, false);

            for (long s = 0; s < slicesl; s++) {
                long idx1 = s * twoSliceStride;
                for (long c = 0; c < columnsl; c++) {
                    long idx2 = 2 * c;
                    for (long r = 0; r < rowsl; r++) {
                        long idx4 = 2 * r;
                        long idx3 = idx1 + r * twoRowStride + idx2;
                        temp.setDouble(idx4, a.getDouble(idx3));
                        temp.setDouble(idx4 + 1, a.getDouble(idx3 + 1));
                    }
                    fftRows.complexInverse(temp, scale);
                    for (long r = 0; r < rowsl; r++) {
                        long idx4 = 2 * r;
                        long idx3 = idx1 + r * twoRowStride + idx2;
                        a.setDouble(idx3, temp.getDouble(idx4));
                        a.setDouble(idx3 + 1, temp.getDouble(idx4 + 1));
                    }
                }
            }

            temp = new DoubleLargeArray(2 * slicesl, false);

            for (long r = 0; r < ldimn2; r++) {
                long idx3 = r * twoRowStride;
                for (long c = 0; c < columnsl; c++) {
                    long idx1 = 2 * c;
                    for (long s = 0; s < slicesl; s++) {
                        long idx2 = 2 * s;
                        long idx4 = s * twoSliceStride + idx3 + idx1;
                        temp.setDouble(idx2, a.getDouble(idx4));
                        temp.setDouble(idx2 + 1, a.getDouble(idx4 + 1));
                    }
                    fftSlices.complexInverse(temp, scale);
                    for (long s = 0; s < slicesl; s++) {
                        long idx2 = 2 * s;
                        long idx4 = s * twoSliceStride + idx3 + idx1;
                        a.setDouble(idx4, temp.getDouble(idx2));
                        a.setDouble(idx4 + 1, temp.getDouble(idx2 + 1));
                    }
                }
            }

            for (long s = 0; s < slicesl; s++) {
                long idx2 = (slicesl - s) % slicesl;
                long idx5 = idx2 * twoSliceStride;
                long idx6 = s * twoSliceStride;
                for (long r = 1; r < n2d2; r++) {
                    long idx4 = rowsl - r;
                    long idx7 = idx4 * twoRowStride;
                    long idx8 = r * twoRowStride;
                    long idx9 = idx5 + idx7;
                    for (long c = 0; c < columnsl; c++) {
                        long idx1 = 2 * c;
                        long idx3 = twon3 - idx1;
                        long idx10 = idx6 + idx8 + idx1;
                        a.setDouble(idx9 + idx3 % twon3, a.getDouble(idx10));
                        a.setDouble(idx9 + (idx3 + 1) % twon3, -a.getDouble(idx10 + 1));
                    }
                }
            }

        }
    }

    private void xdft3da_sub1(int icr, int isgn, double[] a, boolean scale)
    {
        int idx0, idx1, idx2, idx3, idx4, idx5;
        int nt = slices;
        if (nt < rows) {
            nt = rows;
        }
        nt *= 8;
        if (columns == 4) {
            nt >>= 1;
        } else if (columns < 4) {
            nt >>= 2;
        }
        double[] t = new double[nt];
        if (isgn == -1) {
            for (int s = 0; s < slices; s++) {
                idx0 = s * sliceStride;
                if (icr == 0) {
                    for (int r = 0; r < rows; r++) {
                        fftColumns.complexForward(a, idx0 + r * rowStride);
                    }
                } else {
                    for (int r = 0; r < rows; r++) {
                        fftColumns.realForward(a, idx0 + r * rowStride);
                    }
                }
                if (columns > 4) {
                    for (int c = 0; c < columns; c += 8) {
                        for (int r = 0; r < rows; r++) {
                            idx1 = idx0 + r * rowStride + c;
                            idx2 = 2 * r;
                            idx3 = 2 * rows + 2 * r;
                            idx4 = idx3 + 2 * rows;
                            idx5 = idx4 + 2 * rows;
                            t[idx2] = a[idx1];
                            t[idx2 + 1] = a[idx1 + 1];
                            t[idx3] = a[idx1 + 2];
                            t[idx3 + 1] = a[idx1 + 3];
                            t[idx4] = a[idx1 + 4];
                            t[idx4 + 1] = a[idx1 + 5];
                            t[idx5] = a[idx1 + 6];
                            t[idx5 + 1] = a[idx1 + 7];
                        }
                        fftRows.complexForward(t, 0);
                        fftRows.complexForward(t, 2 * rows);
                        fftRows.complexForward(t, 4 * rows);
                        fftRows.complexForward(t, 6 * rows);
                        for (int r = 0; r < rows; r++) {
                            idx1 = idx0 + r * rowStride + c;
                            idx2 = 2 * r;
                            idx3 = 2 * rows + 2 * r;
                            idx4 = idx3 + 2 * rows;
                            idx5 = idx4 + 2 * rows;
                            a[idx1] = t[idx2];
                            a[idx1 + 1] = t[idx2 + 1];
                            a[idx1 + 2] = t[idx3];
                            a[idx1 + 3] = t[idx3 + 1];
                            a[idx1 + 4] = t[idx4];
                            a[idx1 + 5] = t[idx4 + 1];
                            a[idx1 + 6] = t[idx5];
                            a[idx1 + 7] = t[idx5 + 1];
                        }
                    }
                } else if (columns == 4) {
                    for (int r = 0; r < rows; r++) {
                        idx1 = idx0 + r * rowStride;
                        idx2 = 2 * r;
                        idx3 = 2 * rows + 2 * r;
                        t[idx2] = a[idx1];
                        t[idx2 + 1] = a[idx1 + 1];
                        t[idx3] = a[idx1 + 2];
                        t[idx3 + 1] = a[idx1 + 3];
                    }
                    fftRows.complexForward(t, 0);
                    fftRows.complexForward(t, 2 * rows);
                    for (int r = 0; r < rows; r++) {
                        idx1 = idx0 + r * rowStride;
                        idx2 = 2 * r;
                        idx3 = 2 * rows + 2 * r;
                        a[idx1] = t[idx2];
                        a[idx1 + 1] = t[idx2 + 1];
                        a[idx1 + 2] = t[idx3];
                        a[idx1 + 3] = t[idx3 + 1];
                    }
                } else if (columns == 2) {
                    for (int r = 0; r < rows; r++) {
                        idx1 = idx0 + r * rowStride;
                        idx2 = 2 * r;
                        t[idx2] = a[idx1];
                        t[idx2 + 1] = a[idx1 + 1];
                    }
                    fftRows.complexForward(t, 0);
                    for (int r = 0; r < rows; r++) {
                        idx1 = idx0 + r * rowStride;
                        idx2 = 2 * r;
                        a[idx1] = t[idx2];
                        a[idx1 + 1] = t[idx2 + 1];
                    }
                }
            }
        } else {
            for (int s = 0; s < slices; s++) {
                idx0 = s * sliceStride;
                if (icr == 0) {
                    for (int r = 0; r < rows; r++) {
                        fftColumns.complexInverse(a, idx0 + r * rowStride, scale);
                    }
                }
                if (columns > 4) {
                    for (int c = 0; c < columns; c += 8) {
                        for (int r = 0; r < rows; r++) {
                            idx1 = idx0 + r * rowStride + c;
                            idx2 = 2 * r;
                            idx3 = 2 * rows + 2 * r;
                            idx4 = idx3 + 2 * rows;
                            idx5 = idx4 + 2 * rows;
                            t[idx2] = a[idx1];
                            t[idx2 + 1] = a[idx1 + 1];
                            t[idx3] = a[idx1 + 2];
                            t[idx3 + 1] = a[idx1 + 3];
                            t[idx4] = a[idx1 + 4];
                            t[idx4 + 1] = a[idx1 + 5];
                            t[idx5] = a[idx1 + 6];
                            t[idx5 + 1] = a[idx1 + 7];
                        }
                        fftRows.complexInverse(t, 0, scale);
                        fftRows.complexInverse(t, 2 * rows, scale);
                        fftRows.complexInverse(t, 4 * rows, scale);
                        fftRows.complexInverse(t, 6 * rows, scale);
                        for (int r = 0; r < rows; r++) {
                            idx1 = idx0 + r * rowStride + c;
                            idx2 = 2 * r;
                            idx3 = 2 * rows + 2 * r;
                            idx4 = idx3 + 2 * rows;
                            idx5 = idx4 + 2 * rows;
                            a[idx1] = t[idx2];
                            a[idx1 + 1] = t[idx2 + 1];
                            a[idx1 + 2] = t[idx3];
                            a[idx1 + 3] = t[idx3 + 1];
                            a[idx1 + 4] = t[idx4];
                            a[idx1 + 5] = t[idx4 + 1];
                            a[idx1 + 6] = t[idx5];
                            a[idx1 + 7] = t[idx5 + 1];
                        }
                    }
                } else if (columns == 4) {
                    for (int r = 0; r < rows; r++) {
                        idx1 = idx0 + r * rowStride;
                        idx2 = 2 * r;
                        idx3 = 2 * rows + 2 * r;
                        t[idx2] = a[idx1];
                        t[idx2 + 1] = a[idx1 + 1];
                        t[idx3] = a[idx1 + 2];
                        t[idx3 + 1] = a[idx1 + 3];
                    }
                    fftRows.complexInverse(t, 0, scale);
                    fftRows.complexInverse(t, 2 * rows, scale);
                    for (int r = 0; r < rows; r++) {
                        idx1 = idx0 + r * rowStride;
                        idx2 = 2 * r;
                        idx3 = 2 * rows + 2 * r;
                        a[idx1] = t[idx2];
                        a[idx1 + 1] = t[idx2 + 1];
                        a[idx1 + 2] = t[idx3];
                        a[idx1 + 3] = t[idx3 + 1];
                    }
                } else if (columns == 2) {
                    for (int r = 0; r < rows; r++) {
                        idx1 = idx0 + r * rowStride;
                        idx2 = 2 * r;
                        t[idx2] = a[idx1];
                        t[idx2 + 1] = a[idx1 + 1];
                    }
                    fftRows.complexInverse(t, 0, scale);
                    for (int r = 0; r < rows; r++) {
                        idx1 = idx0 + r * rowStride;
                        idx2 = 2 * r;
                        a[idx1] = t[idx2];
                        a[idx1 + 1] = t[idx2 + 1];
                    }
                }
                if (icr != 0) {
                    for (int r = 0; r < rows; r++) {
                        fftColumns.realInverse(a, idx0 + r * rowStride, scale);
                    }
                }
            }
        }
    }

    private void xdft3da_sub1(long icr, int isgn, DoubleLargeArray a, boolean scale)
    {
        long idx0, idx1, idx2, idx3, idx4, idx5;
        long nt = slicesl;
        if (nt < rowsl) {
            nt = rowsl;
        }
        nt *= 8;
        if (columnsl == 4) {
            nt >>= 1l;
        } else if (columnsl < 4) {
            nt >>= 2l;
        }
        DoubleLargeArray t = new DoubleLargeArray(nt);
        if (isgn == -1) {
            for (long s = 0; s < slicesl; s++) {
                idx0 = s * sliceStridel;
                if (icr == 0) {
                    for (long r = 0; r < rowsl; r++) {
                        fftColumns.complexForward(a, idx0 + r * rowStridel);
                    }
                } else {
                    for (long r = 0; r < rowsl; r++) {
                        fftColumns.realForward(a, idx0 + r * rowStridel);
                    }
                }
                if (columnsl > 4) {
                    for (long c = 0; c < columnsl; c += 8) {
                        for (long r = 0; r < rowsl; r++) {
                            idx1 = idx0 + r * rowStridel + c;
                            idx2 = 2 * r;
                            idx3 = 2 * rowsl + 2 * r;
                            idx4 = idx3 + 2 * rowsl;
                            idx5 = idx4 + 2 * rowsl;
                            t.setDouble(idx2, a.getDouble(idx1));
                            t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                            t.setDouble(idx3, a.getDouble(idx1 + 2));
                            t.setDouble(idx3 + 1, a.getDouble(idx1 + 3));
                            t.setDouble(idx4, a.getDouble(idx1 + 4));
                            t.setDouble(idx4 + 1, a.getDouble(idx1 + 5));
                            t.setDouble(idx5, a.getDouble(idx1 + 6));
                            t.setDouble(idx5 + 1, a.getDouble(idx1 + 7));
                        }
                        fftRows.complexForward(t, 0);
                        fftRows.complexForward(t, 2 * rowsl);
                        fftRows.complexForward(t, 4 * rowsl);
                        fftRows.complexForward(t, 6 * rowsl);
                        for (long r = 0; r < rowsl; r++) {
                            idx1 = idx0 + r * rowStridel + c;
                            idx2 = 2 * r;
                            idx3 = 2 * rowsl + 2 * r;
                            idx4 = idx3 + 2 * rowsl;
                            idx5 = idx4 + 2 * rowsl;
                            a.setDouble(idx1, t.getDouble(idx2));
                            a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                            a.setDouble(idx1 + 2, t.getDouble(idx3));
                            a.setDouble(idx1 + 3, t.getDouble(idx3 + 1));
                            a.setDouble(idx1 + 4, t.getDouble(idx4));
                            a.setDouble(idx1 + 5, t.getDouble(idx4 + 1));
                            a.setDouble(idx1 + 6, t.getDouble(idx5));
                            a.setDouble(idx1 + 7, t.getDouble(idx5 + 1));
                        }
                    }
                } else if (columnsl == 4) {
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = idx0 + r * rowStridel;
                        idx2 = 2 * r;
                        idx3 = 2 * rowsl + 2 * r;
                        t.setDouble(idx2, a.getDouble(idx1));
                        t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                        t.setDouble(idx3, a.getDouble(idx1 + 2));
                        t.setDouble(idx3 + 1, a.getDouble(idx1 + 3));
                    }
                    fftRows.complexForward(t, 0);
                    fftRows.complexForward(t, 2 * rowsl);
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = idx0 + r * rowStridel;
                        idx2 = 2 * r;
                        idx3 = 2 * rowsl + 2 * r;
                        a.setDouble(idx1, t.getDouble(idx2));
                        a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                        a.setDouble(idx1 + 2, t.getDouble(idx3));
                        a.setDouble(idx1 + 3, t.getDouble(idx3 + 1));
                    }
                } else if (columnsl == 2) {
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = idx0 + r * rowStridel;
                        idx2 = 2 * r;
                        t.setDouble(idx2, a.getDouble(idx1));
                        t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                    }
                    fftRows.complexForward(t, 0);
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = idx0 + r * rowStridel;
                        idx2 = 2 * r;
                        a.setDouble(idx1, t.getDouble(idx2));
                        a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                    }
                }
            }
        } else {
            for (long s = 0; s < slicesl; s++) {
                idx0 = s * sliceStridel;
                if (icr == 0) {
                    for (long r = 0; r < rowsl; r++) {
                        fftColumns.complexInverse(a, idx0 + r * rowStridel, scale);
                    }
                }
                if (columnsl > 4) {
                    for (long c = 0; c < columnsl; c += 8) {
                        for (long r = 0; r < rowsl; r++) {
                            idx1 = idx0 + r * rowStridel + c;
                            idx2 = 2 * r;
                            idx3 = 2 * rowsl + 2 * r;
                            idx4 = idx3 + 2 * rowsl;
                            idx5 = idx4 + 2 * rowsl;
                            t.setDouble(idx2, a.getDouble(idx1));
                            t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                            t.setDouble(idx3, a.getDouble(idx1 + 2));
                            t.setDouble(idx3 + 1, a.getDouble(idx1 + 3));
                            t.setDouble(idx4, a.getDouble(idx1 + 4));
                            t.setDouble(idx4 + 1, a.getDouble(idx1 + 5));
                            t.setDouble(idx5, a.getDouble(idx1 + 6));
                            t.setDouble(idx5 + 1, a.getDouble(idx1 + 7));
                        }
                        fftRows.complexInverse(t, 0, scale);
                        fftRows.complexInverse(t, 2 * rowsl, scale);
                        fftRows.complexInverse(t, 4 * rowsl, scale);
                        fftRows.complexInverse(t, 6 * rowsl, scale);
                        for (long r = 0; r < rowsl; r++) {
                            idx1 = idx0 + r * rowStridel + c;
                            idx2 = 2 * r;
                            idx3 = 2 * rowsl + 2 * r;
                            idx4 = idx3 + 2 * rowsl;
                            idx5 = idx4 + 2 * rowsl;
                            a.setDouble(idx1, t.getDouble(idx2));
                            a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                            a.setDouble(idx1 + 2, t.getDouble(idx3));
                            a.setDouble(idx1 + 3, t.getDouble(idx3 + 1));
                            a.setDouble(idx1 + 4, t.getDouble(idx4));
                            a.setDouble(idx1 + 5, t.getDouble(idx4 + 1));
                            a.setDouble(idx1 + 6, t.getDouble(idx5));
                            a.setDouble(idx1 + 7, t.getDouble(idx5 + 1));
                        }
                    }
                } else if (columnsl == 4) {
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = idx0 + r * rowStridel;
                        idx2 = 2 * r;
                        idx3 = 2 * rowsl + 2 * r;
                        t.setDouble(idx2, a.getDouble(idx1));
                        t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                        t.setDouble(idx3, a.getDouble(idx1 + 2));
                        t.setDouble(idx3 + 1, a.getDouble(idx1 + 3));
                    }
                    fftRows.complexInverse(t, 0, scale);
                    fftRows.complexInverse(t, 2 * rowsl, scale);
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = idx0 + r * rowStridel;
                        idx2 = 2 * r;
                        idx3 = 2 * rowsl + 2 * r;
                        a.setDouble(idx1, t.getDouble(idx2));
                        a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                        a.setDouble(idx1 + 2, t.getDouble(idx3));
                        a.setDouble(idx1 + 3, t.getDouble(idx3 + 1));
                    }
                } else if (columnsl == 2) {
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = idx0 + r * rowStridel;
                        idx2 = 2 * r;
                        t.setDouble(idx2, a.getDouble(idx1));
                        t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                    }
                    fftRows.complexInverse(t, 0, scale);
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = idx0 + r * rowStridel;
                        idx2 = 2 * r;
                        a.setDouble(idx1, t.getDouble(idx2));
                        a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                    }
                }
                if (icr != 0) {
                    for (long r = 0; r < rowsl; r++) {
                        fftColumns.realInverse(a, idx0 + r * rowStridel, scale);
                    }
                }
            }
        }
    }

    private void xdft3da_sub2(int icr, int isgn, double[] a, boolean scale)
    {
        int idx0, idx1, idx2, idx3, idx4, idx5;
        int nt = slices;
        if (nt < rows) {
            nt = rows;
        }
        nt *= 8;
        if (columns == 4) {
            nt >>= 1;
        } else if (columns < 4) {
            nt >>= 2;
        }
        double[] t = new double[nt];
        if (isgn == -1) {
            for (int s = 0; s < slices; s++) {
                idx0 = s * sliceStride;
                if (icr == 0) {
                    for (int r = 0; r < rows; r++) {
                        fftColumns.complexForward(a, idx0 + r * rowStride);
                    }
                } else {
                    for (int r = 0; r < rows; r++) {
                        fftColumns.realForward(a, idx0 + r * rowStride);
                    }
                }
                if (columns > 4) {
                    for (int c = 0; c < columns; c += 8) {
                        for (int r = 0; r < rows; r++) {
                            idx1 = idx0 + r * rowStride + c;
                            idx2 = 2 * r;
                            idx3 = 2 * rows + 2 * r;
                            idx4 = idx3 + 2 * rows;
                            idx5 = idx4 + 2 * rows;
                            t[idx2] = a[idx1];
                            t[idx2 + 1] = a[idx1 + 1];
                            t[idx3] = a[idx1 + 2];
                            t[idx3 + 1] = a[idx1 + 3];
                            t[idx4] = a[idx1 + 4];
                            t[idx4 + 1] = a[idx1 + 5];
                            t[idx5] = a[idx1 + 6];
                            t[idx5 + 1] = a[idx1 + 7];
                        }
                        fftRows.complexForward(t, 0);
                        fftRows.complexForward(t, 2 * rows);
                        fftRows.complexForward(t, 4 * rows);
                        fftRows.complexForward(t, 6 * rows);
                        for (int r = 0; r < rows; r++) {
                            idx1 = idx0 + r * rowStride + c;
                            idx2 = 2 * r;
                            idx3 = 2 * rows + 2 * r;
                            idx4 = idx3 + 2 * rows;
                            idx5 = idx4 + 2 * rows;
                            a[idx1] = t[idx2];
                            a[idx1 + 1] = t[idx2 + 1];
                            a[idx1 + 2] = t[idx3];
                            a[idx1 + 3] = t[idx3 + 1];
                            a[idx1 + 4] = t[idx4];
                            a[idx1 + 5] = t[idx4 + 1];
                            a[idx1 + 6] = t[idx5];
                            a[idx1 + 7] = t[idx5 + 1];
                        }
                    }
                } else if (columns == 4) {
                    for (int r = 0; r < rows; r++) {
                        idx1 = idx0 + r * rowStride;
                        idx2 = 2 * r;
                        idx3 = 2 * rows + 2 * r;
                        t[idx2] = a[idx1];
                        t[idx2 + 1] = a[idx1 + 1];
                        t[idx3] = a[idx1 + 2];
                        t[idx3 + 1] = a[idx1 + 3];
                    }
                    fftRows.complexForward(t, 0);
                    fftRows.complexForward(t, 2 * rows);
                    for (int r = 0; r < rows; r++) {
                        idx1 = idx0 + r * rowStride;
                        idx2 = 2 * r;
                        idx3 = 2 * rows + 2 * r;
                        a[idx1] = t[idx2];
                        a[idx1 + 1] = t[idx2 + 1];
                        a[idx1 + 2] = t[idx3];
                        a[idx1 + 3] = t[idx3 + 1];
                    }
                } else if (columns == 2) {
                    for (int r = 0; r < rows; r++) {
                        idx1 = idx0 + r * rowStride;
                        idx2 = 2 * r;
                        t[idx2] = a[idx1];
                        t[idx2 + 1] = a[idx1 + 1];
                    }
                    fftRows.complexForward(t, 0);
                    for (int r = 0; r < rows; r++) {
                        idx1 = idx0 + r * rowStride;
                        idx2 = 2 * r;
                        a[idx1] = t[idx2];
                        a[idx1 + 1] = t[idx2 + 1];
                    }
                }
            }
        } else {
            for (int s = 0; s < slices; s++) {
                idx0 = s * sliceStride;
                if (icr == 0) {
                    for (int r = 0; r < rows; r++) {
                        fftColumns.complexInverse(a, idx0 + r * rowStride, scale);
                    }
                } else {
                    for (int r = 0; r < rows; r++) {
                        fftColumns.realInverse2(a, idx0 + r * rowStride, scale);
                    }
                }
                if (columns > 4) {
                    for (int c = 0; c < columns; c += 8) {
                        for (int r = 0; r < rows; r++) {
                            idx1 = idx0 + r * rowStride + c;
                            idx2 = 2 * r;
                            idx3 = 2 * rows + 2 * r;
                            idx4 = idx3 + 2 * rows;
                            idx5 = idx4 + 2 * rows;
                            t[idx2] = a[idx1];
                            t[idx2 + 1] = a[idx1 + 1];
                            t[idx3] = a[idx1 + 2];
                            t[idx3 + 1] = a[idx1 + 3];
                            t[idx4] = a[idx1 + 4];
                            t[idx4 + 1] = a[idx1 + 5];
                            t[idx5] = a[idx1 + 6];
                            t[idx5 + 1] = a[idx1 + 7];
                        }
                        fftRows.complexInverse(t, 0, scale);
                        fftRows.complexInverse(t, 2 * rows, scale);
                        fftRows.complexInverse(t, 4 * rows, scale);
                        fftRows.complexInverse(t, 6 * rows, scale);
                        for (int r = 0; r < rows; r++) {
                            idx1 = idx0 + r * rowStride + c;
                            idx2 = 2 * r;
                            idx3 = 2 * rows + 2 * r;
                            idx4 = idx3 + 2 * rows;
                            idx5 = idx4 + 2 * rows;
                            a[idx1] = t[idx2];
                            a[idx1 + 1] = t[idx2 + 1];
                            a[idx1 + 2] = t[idx3];
                            a[idx1 + 3] = t[idx3 + 1];
                            a[idx1 + 4] = t[idx4];
                            a[idx1 + 5] = t[idx4 + 1];
                            a[idx1 + 6] = t[idx5];
                            a[idx1 + 7] = t[idx5 + 1];
                        }
                    }
                } else if (columns == 4) {
                    for (int r = 0; r < rows; r++) {
                        idx1 = idx0 + r * rowStride;
                        idx2 = 2 * r;
                        idx3 = 2 * rows + 2 * r;
                        t[idx2] = a[idx1];
                        t[idx2 + 1] = a[idx1 + 1];
                        t[idx3] = a[idx1 + 2];
                        t[idx3 + 1] = a[idx1 + 3];
                    }
                    fftRows.complexInverse(t, 0, scale);
                    fftRows.complexInverse(t, 2 * rows, scale);
                    for (int r = 0; r < rows; r++) {
                        idx1 = idx0 + r * rowStride;
                        idx2 = 2 * r;
                        idx3 = 2 * rows + 2 * r;
                        a[idx1] = t[idx2];
                        a[idx1 + 1] = t[idx2 + 1];
                        a[idx1 + 2] = t[idx3];
                        a[idx1 + 3] = t[idx3 + 1];
                    }
                } else if (columns == 2) {
                    for (int r = 0; r < rows; r++) {
                        idx1 = idx0 + r * rowStride;
                        idx2 = 2 * r;
                        t[idx2] = a[idx1];
                        t[idx2 + 1] = a[idx1 + 1];
                    }
                    fftRows.complexInverse(t, 0, scale);
                    for (int r = 0; r < rows; r++) {
                        idx1 = idx0 + r * rowStride;
                        idx2 = 2 * r;
                        a[idx1] = t[idx2];
                        a[idx1 + 1] = t[idx2 + 1];
                    }
                }
            }
        }
    }

    private void xdft3da_sub2(long icr, int isgn, DoubleLargeArray a, boolean scale)
    {
        long idx0, idx1, idx2, idx3, idx4, idx5;
        long nt = slicesl;
        if (nt < rowsl) {
            nt = rowsl;
        }
        nt *= 8;
        if (columnsl == 4) {
            nt >>= 1;
        } else if (columnsl < 4) {
            nt >>= 2;
        }
        DoubleLargeArray t = new DoubleLargeArray(nt);
        if (isgn == -1) {
            for (long s = 0; s < slicesl; s++) {
                idx0 = s * sliceStridel;
                if (icr == 0) {
                    for (long r = 0; r < rowsl; r++) {
                        fftColumns.complexForward(a, idx0 + r * rowStridel);
                    }
                } else {
                    for (long r = 0; r < rowsl; r++) {
                        fftColumns.realForward(a, idx0 + r * rowStridel);
                    }
                }
                if (columnsl > 4) {
                    for (long c = 0; c < columnsl; c += 8) {
                        for (long r = 0; r < rowsl; r++) {
                            idx1 = idx0 + r * rowStridel + c;
                            idx2 = 2 * r;
                            idx3 = 2 * rowsl + 2 * r;
                            idx4 = idx3 + 2 * rowsl;
                            idx5 = idx4 + 2 * rowsl;
                            t.setDouble(idx2, a.getDouble(idx1));
                            t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                            t.setDouble(idx3, a.getDouble(idx1 + 2));
                            t.setDouble(idx3 + 1, a.getDouble(idx1 + 3));
                            t.setDouble(idx4, a.getDouble(idx1 + 4));
                            t.setDouble(idx4 + 1, a.getDouble(idx1 + 5));
                            t.setDouble(idx5, a.getDouble(idx1 + 6));
                            t.setDouble(idx5 + 1, a.getDouble(idx1 + 7));
                        }
                        fftRows.complexForward(t, 0);
                        fftRows.complexForward(t, 2 * rowsl);
                        fftRows.complexForward(t, 4 * rowsl);
                        fftRows.complexForward(t, 6 * rowsl);
                        for (long r = 0; r < rowsl; r++) {
                            idx1 = idx0 + r * rowStridel + c;
                            idx2 = 2 * r;
                            idx3 = 2 * rowsl + 2 * r;
                            idx4 = idx3 + 2 * rowsl;
                            idx5 = idx4 + 2 * rowsl;
                            a.setDouble(idx1, t.getDouble(idx2));
                            a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                            a.setDouble(idx1 + 2, t.getDouble(idx3));
                            a.setDouble(idx1 + 3, t.getDouble(idx3 + 1));
                            a.setDouble(idx1 + 4, t.getDouble(idx4));
                            a.setDouble(idx1 + 5, t.getDouble(idx4 + 1));
                            a.setDouble(idx1 + 6, t.getDouble(idx5));
                            a.setDouble(idx1 + 7, t.getDouble(idx5 + 1));
                        }
                    }
                } else if (columnsl == 4) {
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = idx0 + r * rowStridel;
                        idx2 = 2 * r;
                        idx3 = 2 * rowsl + 2 * r;
                        t.setDouble(idx2, a.getDouble(idx1));
                        t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                        t.setDouble(idx3, a.getDouble(idx1 + 2));
                        t.setDouble(idx3 + 1, a.getDouble(idx1 + 3));
                    }
                    fftRows.complexForward(t, 0);
                    fftRows.complexForward(t, 2 * rowsl);
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = idx0 + r * rowStridel;
                        idx2 = 2 * r;
                        idx3 = 2 * rowsl + 2 * r;
                        a.setDouble(idx1, t.getDouble(idx2));
                        a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                        a.setDouble(idx1 + 2, t.getDouble(idx3));
                        a.setDouble(idx1 + 3, t.getDouble(idx3 + 1));
                    }
                } else if (columnsl == 2) {
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = idx0 + r * rowStridel;
                        idx2 = 2 * r;
                        t.setDouble(idx2, a.getDouble(idx1));
                        t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                    }
                    fftRows.complexForward(t, 0);
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = idx0 + r * rowStridel;
                        idx2 = 2 * r;
                        a.setDouble(idx1, t.getDouble(idx2));
                        a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                    }
                }
            }
        } else {
            for (long s = 0; s < slicesl; s++) {
                idx0 = s * sliceStridel;
                if (icr == 0) {
                    for (long r = 0; r < rowsl; r++) {
                        fftColumns.complexInverse(a, idx0 + r * rowStridel, scale);
                    }
                } else {
                    for (long r = 0; r < rowsl; r++) {
                        fftColumns.realInverse2(a, idx0 + r * rowStridel, scale);
                    }
                }
                if (columnsl > 4) {
                    for (long c = 0; c < columnsl; c += 8) {
                        for (long r = 0; r < rowsl; r++) {
                            idx1 = idx0 + r * rowStridel + c;
                            idx2 = 2 * r;
                            idx3 = 2 * rowsl + 2 * r;
                            idx4 = idx3 + 2 * rowsl;
                            idx5 = idx4 + 2 * rowsl;
                            t.setDouble(idx2, a.getDouble(idx1));
                            t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                            t.setDouble(idx3, a.getDouble(idx1 + 2));
                            t.setDouble(idx3 + 1, a.getDouble(idx1 + 3));
                            t.setDouble(idx4, a.getDouble(idx1 + 4));
                            t.setDouble(idx4 + 1, a.getDouble(idx1 + 5));
                            t.setDouble(idx5, a.getDouble(idx1 + 6));
                            t.setDouble(idx5 + 1, a.getDouble(idx1 + 7));
                        }
                        fftRows.complexInverse(t, 0, scale);
                        fftRows.complexInverse(t, 2 * rowsl, scale);
                        fftRows.complexInverse(t, 4 * rowsl, scale);
                        fftRows.complexInverse(t, 6 * rowsl, scale);
                        for (long r = 0; r < rowsl; r++) {
                            idx1 = idx0 + r * rowStridel + c;
                            idx2 = 2 * r;
                            idx3 = 2 * rowsl + 2 * r;
                            idx4 = idx3 + 2 * rowsl;
                            idx5 = idx4 + 2 * rowsl;
                            a.setDouble(idx1, t.getDouble(idx2));
                            a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                            a.setDouble(idx1 + 2, t.getDouble(idx3));
                            a.setDouble(idx1 + 3, t.getDouble(idx3 + 1));
                            a.setDouble(idx1 + 4, t.getDouble(idx4));
                            a.setDouble(idx1 + 5, t.getDouble(idx4 + 1));
                            a.setDouble(idx1 + 6, t.getDouble(idx5));
                            a.setDouble(idx1 + 7, t.getDouble(idx5 + 1));
                        }
                    }
                } else if (columnsl == 4) {
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = idx0 + r * rowStridel;
                        idx2 = 2 * r;
                        idx3 = 2 * rowsl + 2 * r;
                        t.setDouble(idx2, a.getDouble(idx1));
                        t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                        t.setDouble(idx3, a.getDouble(idx1 + 2));
                        t.setDouble(idx3 + 1, a.getDouble(idx1 + 3));
                    }
                    fftRows.complexInverse(t, 0, scale);
                    fftRows.complexInverse(t, 2 * rowsl, scale);
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = idx0 + r * rowStridel;
                        idx2 = 2 * r;
                        idx3 = 2 * rowsl + 2 * r;
                        a.setDouble(idx1, t.getDouble(idx2));
                        a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                        a.setDouble(idx1 + 2, t.getDouble(idx3));
                        a.setDouble(idx1 + 3, t.getDouble(idx3 + 1));
                    }
                } else if (columnsl == 2) {
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = idx0 + r * rowStridel;
                        idx2 = 2 * r;
                        t.setDouble(idx2, a.getDouble(idx1));
                        t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                    }
                    fftRows.complexInverse(t, 0, scale);
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = idx0 + r * rowStridel;
                        idx2 = 2 * r;
                        a.setDouble(idx1, t.getDouble(idx2));
                        a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                    }
                }
            }
        }
    }

    private void xdft3da_sub1(int icr, int isgn, double[][][] a, boolean scale)
    {
        int idx2, idx3, idx4, idx5;
        int nt = slices;
        if (nt < rows) {
            nt = rows;
        }
        nt *= 8;
        if (columns == 4) {
            nt >>= 1;
        } else if (columns < 4) {
            nt >>= 2;
        }
        double[] t = new double[nt];
        if (isgn == -1) {
            for (int s = 0; s < slices; s++) {
                if (icr == 0) {
                    for (int r = 0; r < rows; r++) {
                        fftColumns.complexForward(a[s][r]);
                    }
                } else {
                    for (int r = 0; r < rows; r++) {
                        fftColumns.realForward(a[s][r], 0);
                    }
                }
                if (columns > 4) {
                    for (int c = 0; c < columns; c += 8) {
                        for (int r = 0; r < rows; r++) {
                            idx2 = 2 * r;
                            idx3 = 2 * rows + 2 * r;
                            idx4 = idx3 + 2 * rows;
                            idx5 = idx4 + 2 * rows;
                            t[idx2] = a[s][r][c];
                            t[idx2 + 1] = a[s][r][c + 1];
                            t[idx3] = a[s][r][c + 2];
                            t[idx3 + 1] = a[s][r][c + 3];
                            t[idx4] = a[s][r][c + 4];
                            t[idx4 + 1] = a[s][r][c + 5];
                            t[idx5] = a[s][r][c + 6];
                            t[idx5 + 1] = a[s][r][c + 7];
                        }
                        fftRows.complexForward(t, 0);
                        fftRows.complexForward(t, 2 * rows);
                        fftRows.complexForward(t, 4 * rows);
                        fftRows.complexForward(t, 6 * rows);
                        for (int r = 0; r < rows; r++) {
                            idx2 = 2 * r;
                            idx3 = 2 * rows + 2 * r;
                            idx4 = idx3 + 2 * rows;
                            idx5 = idx4 + 2 * rows;
                            a[s][r][c] = t[idx2];
                            a[s][r][c + 1] = t[idx2 + 1];
                            a[s][r][c + 2] = t[idx3];
                            a[s][r][c + 3] = t[idx3 + 1];
                            a[s][r][c + 4] = t[idx4];
                            a[s][r][c + 5] = t[idx4 + 1];
                            a[s][r][c + 6] = t[idx5];
                            a[s][r][c + 7] = t[idx5 + 1];
                        }
                    }
                } else if (columns == 4) {
                    for (int r = 0; r < rows; r++) {
                        idx2 = 2 * r;
                        idx3 = 2 * rows + 2 * r;
                        t[idx2] = a[s][r][0];
                        t[idx2 + 1] = a[s][r][1];
                        t[idx3] = a[s][r][2];
                        t[idx3 + 1] = a[s][r][3];
                    }
                    fftRows.complexForward(t, 0);
                    fftRows.complexForward(t, 2 * rows);
                    for (int r = 0; r < rows; r++) {
                        idx2 = 2 * r;
                        idx3 = 2 * rows + 2 * r;
                        a[s][r][0] = t[idx2];
                        a[s][r][1] = t[idx2 + 1];
                        a[s][r][2] = t[idx3];
                        a[s][r][3] = t[idx3 + 1];
                    }
                } else if (columns == 2) {
                    for (int r = 0; r < rows; r++) {
                        idx2 = 2 * r;
                        t[idx2] = a[s][r][0];
                        t[idx2 + 1] = a[s][r][1];
                    }
                    fftRows.complexForward(t, 0);
                    for (int r = 0; r < rows; r++) {
                        idx2 = 2 * r;
                        a[s][r][0] = t[idx2];
                        a[s][r][1] = t[idx2 + 1];
                    }
                }
            }
        } else {
            for (int s = 0; s < slices; s++) {
                if (icr == 0) {
                    for (int r = 0; r < rows; r++) {
                        fftColumns.complexInverse(a[s][r], scale);
                    }
                }
                if (columns > 4) {
                    for (int c = 0; c < columns; c += 8) {
                        for (int r = 0; r < rows; r++) {
                            idx2 = 2 * r;
                            idx3 = 2 * rows + 2 * r;
                            idx4 = idx3 + 2 * rows;
                            idx5 = idx4 + 2 * rows;
                            t[idx2] = a[s][r][c];
                            t[idx2 + 1] = a[s][r][c + 1];
                            t[idx3] = a[s][r][c + 2];
                            t[idx3 + 1] = a[s][r][c + 3];
                            t[idx4] = a[s][r][c + 4];
                            t[idx4 + 1] = a[s][r][c + 5];
                            t[idx5] = a[s][r][c + 6];
                            t[idx5 + 1] = a[s][r][c + 7];
                        }
                        fftRows.complexInverse(t, 0, scale);
                        fftRows.complexInverse(t, 2 * rows, scale);
                        fftRows.complexInverse(t, 4 * rows, scale);
                        fftRows.complexInverse(t, 6 * rows, scale);
                        for (int r = 0; r < rows; r++) {
                            idx2 = 2 * r;
                            idx3 = 2 * rows + 2 * r;
                            idx4 = idx3 + 2 * rows;
                            idx5 = idx4 + 2 * rows;
                            a[s][r][c] = t[idx2];
                            a[s][r][c + 1] = t[idx2 + 1];
                            a[s][r][c + 2] = t[idx3];
                            a[s][r][c + 3] = t[idx3 + 1];
                            a[s][r][c + 4] = t[idx4];
                            a[s][r][c + 5] = t[idx4 + 1];
                            a[s][r][c + 6] = t[idx5];
                            a[s][r][c + 7] = t[idx5 + 1];
                        }
                    }
                } else if (columns == 4) {
                    for (int r = 0; r < rows; r++) {
                        idx2 = 2 * r;
                        idx3 = 2 * rows + 2 * r;
                        t[idx2] = a[s][r][0];
                        t[idx2 + 1] = a[s][r][1];
                        t[idx3] = a[s][r][2];
                        t[idx3 + 1] = a[s][r][3];
                    }
                    fftRows.complexInverse(t, 0, scale);
                    fftRows.complexInverse(t, 2 * rows, scale);
                    for (int r = 0; r < rows; r++) {
                        idx2 = 2 * r;
                        idx3 = 2 * rows + 2 * r;
                        a[s][r][0] = t[idx2];
                        a[s][r][1] = t[idx2 + 1];
                        a[s][r][2] = t[idx3];
                        a[s][r][3] = t[idx3 + 1];
                    }
                } else if (columns == 2) {
                    for (int r = 0; r < rows; r++) {
                        idx2 = 2 * r;
                        t[idx2] = a[s][r][0];
                        t[idx2 + 1] = a[s][r][1];
                    }
                    fftRows.complexInverse(t, 0, scale);
                    for (int r = 0; r < rows; r++) {
                        idx2 = 2 * r;
                        a[s][r][0] = t[idx2];
                        a[s][r][1] = t[idx2 + 1];
                    }
                }
                if (icr != 0) {
                    for (int r = 0; r < rows; r++) {
                        fftColumns.realInverse(a[s][r], scale);
                    }
                }
            }
        }
    }

    private void xdft3da_sub2(int icr, int isgn, double[][][] a, boolean scale)
    {
        int idx2, idx3, idx4, idx5;
        int nt = slices;
        if (nt < rows) {
            nt = rows;
        }
        nt *= 8;
        if (columns == 4) {
            nt >>= 1;
        } else if (columns < 4) {
            nt >>= 2;
        }
        double[] t = new double[nt];
        if (isgn == -1) {
            for (int s = 0; s < slices; s++) {
                if (icr == 0) {
                    for (int r = 0; r < rows; r++) {
                        fftColumns.complexForward(a[s][r]);
                    }
                } else {
                    for (int r = 0; r < rows; r++) {
                        fftColumns.realForward(a[s][r]);
                    }
                }
                if (columns > 4) {
                    for (int c = 0; c < columns; c += 8) {
                        for (int r = 0; r < rows; r++) {
                            idx2 = 2 * r;
                            idx3 = 2 * rows + 2 * r;
                            idx4 = idx3 + 2 * rows;
                            idx5 = idx4 + 2 * rows;
                            t[idx2] = a[s][r][c];
                            t[idx2 + 1] = a[s][r][c + 1];
                            t[idx3] = a[s][r][c + 2];
                            t[idx3 + 1] = a[s][r][c + 3];
                            t[idx4] = a[s][r][c + 4];
                            t[idx4 + 1] = a[s][r][c + 5];
                            t[idx5] = a[s][r][c + 6];
                            t[idx5 + 1] = a[s][r][c + 7];
                        }
                        fftRows.complexForward(t, 0);
                        fftRows.complexForward(t, 2 * rows);
                        fftRows.complexForward(t, 4 * rows);
                        fftRows.complexForward(t, 6 * rows);
                        for (int r = 0; r < rows; r++) {
                            idx2 = 2 * r;
                            idx3 = 2 * rows + 2 * r;
                            idx4 = idx3 + 2 * rows;
                            idx5 = idx4 + 2 * rows;
                            a[s][r][c] = t[idx2];
                            a[s][r][c + 1] = t[idx2 + 1];
                            a[s][r][c + 2] = t[idx3];
                            a[s][r][c + 3] = t[idx3 + 1];
                            a[s][r][c + 4] = t[idx4];
                            a[s][r][c + 5] = t[idx4 + 1];
                            a[s][r][c + 6] = t[idx5];
                            a[s][r][c + 7] = t[idx5 + 1];
                        }
                    }
                } else if (columns == 4) {
                    for (int r = 0; r < rows; r++) {
                        idx2 = 2 * r;
                        idx3 = 2 * rows + 2 * r;
                        t[idx2] = a[s][r][0];
                        t[idx2 + 1] = a[s][r][1];
                        t[idx3] = a[s][r][2];
                        t[idx3 + 1] = a[s][r][3];
                    }
                    fftRows.complexForward(t, 0);
                    fftRows.complexForward(t, 2 * rows);
                    for (int r = 0; r < rows; r++) {
                        idx2 = 2 * r;
                        idx3 = 2 * rows + 2 * r;
                        a[s][r][0] = t[idx2];
                        a[s][r][1] = t[idx2 + 1];
                        a[s][r][2] = t[idx3];
                        a[s][r][3] = t[idx3 + 1];
                    }
                } else if (columns == 2) {
                    for (int r = 0; r < rows; r++) {
                        idx2 = 2 * r;
                        t[idx2] = a[s][r][0];
                        t[idx2 + 1] = a[s][r][1];
                    }
                    fftRows.complexForward(t, 0);
                    for (int r = 0; r < rows; r++) {
                        idx2 = 2 * r;
                        a[s][r][0] = t[idx2];
                        a[s][r][1] = t[idx2 + 1];
                    }
                }
            }
        } else {
            for (int s = 0; s < slices; s++) {
                if (icr == 0) {
                    for (int r = 0; r < rows; r++) {
                        fftColumns.complexInverse(a[s][r], scale);
                    }
                } else {
                    for (int r = 0; r < rows; r++) {
                        fftColumns.realInverse2(a[s][r], 0, scale);
                    }
                }
                if (columns > 4) {
                    for (int c = 0; c < columns; c += 8) {
                        for (int r = 0; r < rows; r++) {
                            idx2 = 2 * r;
                            idx3 = 2 * rows + 2 * r;
                            idx4 = idx3 + 2 * rows;
                            idx5 = idx4 + 2 * rows;
                            t[idx2] = a[s][r][c];
                            t[idx2 + 1] = a[s][r][c + 1];
                            t[idx3] = a[s][r][c + 2];
                            t[idx3 + 1] = a[s][r][c + 3];
                            t[idx4] = a[s][r][c + 4];
                            t[idx4 + 1] = a[s][r][c + 5];
                            t[idx5] = a[s][r][c + 6];
                            t[idx5 + 1] = a[s][r][c + 7];
                        }
                        fftRows.complexInverse(t, 0, scale);
                        fftRows.complexInverse(t, 2 * rows, scale);
                        fftRows.complexInverse(t, 4 * rows, scale);
                        fftRows.complexInverse(t, 6 * rows, scale);
                        for (int r = 0; r < rows; r++) {
                            idx2 = 2 * r;
                            idx3 = 2 * rows + 2 * r;
                            idx4 = idx3 + 2 * rows;
                            idx5 = idx4 + 2 * rows;
                            a[s][r][c] = t[idx2];
                            a[s][r][c + 1] = t[idx2 + 1];
                            a[s][r][c + 2] = t[idx3];
                            a[s][r][c + 3] = t[idx3 + 1];
                            a[s][r][c + 4] = t[idx4];
                            a[s][r][c + 5] = t[idx4 + 1];
                            a[s][r][c + 6] = t[idx5];
                            a[s][r][c + 7] = t[idx5 + 1];
                        }
                    }
                } else if (columns == 4) {
                    for (int r = 0; r < rows; r++) {
                        idx2 = 2 * r;
                        idx3 = 2 * rows + 2 * r;
                        t[idx2] = a[s][r][0];
                        t[idx2 + 1] = a[s][r][1];
                        t[idx3] = a[s][r][2];
                        t[idx3 + 1] = a[s][r][3];
                    }
                    fftRows.complexInverse(t, 0, scale);
                    fftRows.complexInverse(t, 2 * rows, scale);
                    for (int r = 0; r < rows; r++) {
                        idx2 = 2 * r;
                        idx3 = 2 * rows + 2 * r;
                        a[s][r][0] = t[idx2];
                        a[s][r][1] = t[idx2 + 1];
                        a[s][r][2] = t[idx3];
                        a[s][r][3] = t[idx3 + 1];
                    }
                } else if (columns == 2) {
                    for (int r = 0; r < rows; r++) {
                        idx2 = 2 * r;
                        t[idx2] = a[s][r][0];
                        t[idx2 + 1] = a[s][r][1];
                    }
                    fftRows.complexInverse(t, 0, scale);
                    for (int r = 0; r < rows; r++) {
                        idx2 = 2 * r;
                        a[s][r][0] = t[idx2];
                        a[s][r][1] = t[idx2 + 1];
                    }
                }
            }
        }
    }

    private void cdft3db_sub(int isgn, double[] a, boolean scale)
    {
        int idx0, idx1, idx2, idx3, idx4, idx5;
        int nt = slices;
        if (nt < rows) {
            nt = rows;
        }
        nt *= 8;
        if (columns == 4) {
            nt >>= 1;
        } else if (columns < 4) {
            nt >>= 2;
        }
        double[] t = new double[nt];
        if (isgn == -1) {
            if (columns > 4) {
                for (int r = 0; r < rows; r++) {
                    idx0 = r * rowStride;
                    for (int c = 0; c < columns; c += 8) {
                        for (int s = 0; s < slices; s++) {
                            idx1 = s * sliceStride + idx0 + c;
                            idx2 = 2 * s;
                            idx3 = 2 * slices + 2 * s;
                            idx4 = idx3 + 2 * slices;
                            idx5 = idx4 + 2 * slices;
                            t[idx2] = a[idx1];
                            t[idx2 + 1] = a[idx1 + 1];
                            t[idx3] = a[idx1 + 2];
                            t[idx3 + 1] = a[idx1 + 3];
                            t[idx4] = a[idx1 + 4];
                            t[idx4 + 1] = a[idx1 + 5];
                            t[idx5] = a[idx1 + 6];
                            t[idx5 + 1] = a[idx1 + 7];
                        }
                        fftSlices.complexForward(t, 0);
                        fftSlices.complexForward(t, 2 * slices);
                        fftSlices.complexForward(t, 4 * slices);
                        fftSlices.complexForward(t, 6 * slices);
                        for (int s = 0; s < slices; s++) {
                            idx1 = s * sliceStride + idx0 + c;
                            idx2 = 2 * s;
                            idx3 = 2 * slices + 2 * s;
                            idx4 = idx3 + 2 * slices;
                            idx5 = idx4 + 2 * slices;
                            a[idx1] = t[idx2];
                            a[idx1 + 1] = t[idx2 + 1];
                            a[idx1 + 2] = t[idx3];
                            a[idx1 + 3] = t[idx3 + 1];
                            a[idx1 + 4] = t[idx4];
                            a[idx1 + 5] = t[idx4 + 1];
                            a[idx1 + 6] = t[idx5];
                            a[idx1 + 7] = t[idx5 + 1];
                        }
                    }
                }
            } else if (columns == 4) {
                for (int r = 0; r < rows; r++) {
                    idx0 = r * rowStride;
                    for (int s = 0; s < slices; s++) {
                        idx1 = s * sliceStride + idx0;
                        idx2 = 2 * s;
                        idx3 = 2 * slices + 2 * s;
                        t[idx2] = a[idx1];
                        t[idx2 + 1] = a[idx1 + 1];
                        t[idx3] = a[idx1 + 2];
                        t[idx3 + 1] = a[idx1 + 3];
                    }
                    fftSlices.complexForward(t, 0);
                    fftSlices.complexForward(t, 2 * slices);
                    for (int s = 0; s < slices; s++) {
                        idx1 = s * sliceStride + idx0;
                        idx2 = 2 * s;
                        idx3 = 2 * slices + 2 * s;
                        a[idx1] = t[idx2];
                        a[idx1 + 1] = t[idx2 + 1];
                        a[idx1 + 2] = t[idx3];
                        a[idx1 + 3] = t[idx3 + 1];
                    }
                }
            } else if (columns == 2) {
                for (int r = 0; r < rows; r++) {
                    idx0 = r * rowStride;
                    for (int s = 0; s < slices; s++) {
                        idx1 = s * sliceStride + idx0;
                        idx2 = 2 * s;
                        t[idx2] = a[idx1];
                        t[idx2 + 1] = a[idx1 + 1];
                    }
                    fftSlices.complexForward(t, 0);
                    for (int s = 0; s < slices; s++) {
                        idx1 = s * sliceStride + idx0;
                        idx2 = 2 * s;
                        a[idx1] = t[idx2];
                        a[idx1 + 1] = t[idx2 + 1];
                    }
                }
            }
        } else if (columns > 4) {
            for (int r = 0; r < rows; r++) {
                idx0 = r * rowStride;
                for (int c = 0; c < columns; c += 8) {
                    for (int s = 0; s < slices; s++) {
                        idx1 = s * sliceStride + idx0 + c;
                        idx2 = 2 * s;
                        idx3 = 2 * slices + 2 * s;
                        idx4 = idx3 + 2 * slices;
                        idx5 = idx4 + 2 * slices;
                        t[idx2] = a[idx1];
                        t[idx2 + 1] = a[idx1 + 1];
                        t[idx3] = a[idx1 + 2];
                        t[idx3 + 1] = a[idx1 + 3];
                        t[idx4] = a[idx1 + 4];
                        t[idx4 + 1] = a[idx1 + 5];
                        t[idx5] = a[idx1 + 6];
                        t[idx5 + 1] = a[idx1 + 7];
                    }
                    fftSlices.complexInverse(t, 0, scale);
                    fftSlices.complexInverse(t, 2 * slices, scale);
                    fftSlices.complexInverse(t, 4 * slices, scale);
                    fftSlices.complexInverse(t, 6 * slices, scale);
                    for (int s = 0; s < slices; s++) {
                        idx1 = s * sliceStride + idx0 + c;
                        idx2 = 2 * s;
                        idx3 = 2 * slices + 2 * s;
                        idx4 = idx3 + 2 * slices;
                        idx5 = idx4 + 2 * slices;
                        a[idx1] = t[idx2];
                        a[idx1 + 1] = t[idx2 + 1];
                        a[idx1 + 2] = t[idx3];
                        a[idx1 + 3] = t[idx3 + 1];
                        a[idx1 + 4] = t[idx4];
                        a[idx1 + 5] = t[idx4 + 1];
                        a[idx1 + 6] = t[idx5];
                        a[idx1 + 7] = t[idx5 + 1];
                    }
                }
            }
        } else if (columns == 4) {
            for (int r = 0; r < rows; r++) {
                idx0 = r * rowStride;
                for (int s = 0; s < slices; s++) {
                    idx1 = s * sliceStride + idx0;
                    idx2 = 2 * s;
                    idx3 = 2 * slices + 2 * s;
                    t[idx2] = a[idx1];
                    t[idx2 + 1] = a[idx1 + 1];
                    t[idx3] = a[idx1 + 2];
                    t[idx3 + 1] = a[idx1 + 3];
                }
                fftSlices.complexInverse(t, 0, scale);
                fftSlices.complexInverse(t, 2 * slices, scale);
                for (int s = 0; s < slices; s++) {
                    idx1 = s * sliceStride + idx0;
                    idx2 = 2 * s;
                    idx3 = 2 * slices + 2 * s;
                    a[idx1] = t[idx2];
                    a[idx1 + 1] = t[idx2 + 1];
                    a[idx1 + 2] = t[idx3];
                    a[idx1 + 3] = t[idx3 + 1];
                }
            }
        } else if (columns == 2) {
            for (int r = 0; r < rows; r++) {
                idx0 = r * rowStride;
                for (int s = 0; s < slices; s++) {
                    idx1 = s * sliceStride + idx0;
                    idx2 = 2 * s;
                    t[idx2] = a[idx1];
                    t[idx2 + 1] = a[idx1 + 1];
                }
                fftSlices.complexInverse(t, 0, scale);
                for (int s = 0; s < slices; s++) {
                    idx1 = s * sliceStride + idx0;
                    idx2 = 2 * s;
                    a[idx1] = t[idx2];
                    a[idx1 + 1] = t[idx2 + 1];
                }
            }
        }
    }

    private void cdft3db_sub(int isgn, DoubleLargeArray a, boolean scale)
    {
        long idx0, idx1, idx2, idx3, idx4, idx5;
        long nt = slicesl;
        if (nt < rowsl) {
            nt = rowsl;
        }
        nt *= 8;
        if (columnsl == 4) {
            nt >>= 1;
        } else if (columnsl < 4) {
            nt >>= 2;
        }
        DoubleLargeArray t = new DoubleLargeArray(nt);
        if (isgn == -1) {
            if (columnsl > 4) {
                for (long r = 0; r < rowsl; r++) {
                    idx0 = r * rowStridel;
                    for (long c = 0; c < columnsl; c += 8) {
                        for (long s = 0; s < slicesl; s++) {
                            idx1 = s * sliceStridel + idx0 + c;
                            idx2 = 2 * s;
                            idx3 = 2 * slicesl + 2 * s;
                            idx4 = idx3 + 2 * slicesl;
                            idx5 = idx4 + 2 * slicesl;
                            t.setDouble(idx2, a.getDouble(idx1));
                            t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                            t.setDouble(idx3, a.getDouble(idx1 + 2));
                            t.setDouble(idx3 + 1, a.getDouble(idx1 + 3));
                            t.setDouble(idx4, a.getDouble(idx1 + 4));
                            t.setDouble(idx4 + 1, a.getDouble(idx1 + 5));
                            t.setDouble(idx5, a.getDouble(idx1 + 6));
                            t.setDouble(idx5 + 1, a.getDouble(idx1 + 7));
                        }
                        fftSlices.complexForward(t, 0);
                        fftSlices.complexForward(t, 2 * slicesl);
                        fftSlices.complexForward(t, 4 * slicesl);
                        fftSlices.complexForward(t, 6 * slicesl);
                        for (long s = 0; s < slicesl; s++) {
                            idx1 = s * sliceStridel + idx0 + c;
                            idx2 = 2 * s;
                            idx3 = 2 * slicesl + 2 * s;
                            idx4 = idx3 + 2 * slicesl;
                            idx5 = idx4 + 2 * slicesl;
                            a.setDouble(idx1, t.getDouble(idx2));
                            a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                            a.setDouble(idx1 + 2, t.getDouble(idx3));
                            a.setDouble(idx1 + 3, t.getDouble(idx3 + 1));
                            a.setDouble(idx1 + 4, t.getDouble(idx4));
                            a.setDouble(idx1 + 5, t.getDouble(idx4 + 1));
                            a.setDouble(idx1 + 6, t.getDouble(idx5));
                            a.setDouble(idx1 + 7, t.getDouble(idx5 + 1));
                        }
                    }
                }
            } else if (columnsl == 4) {
                for (long r = 0; r < rowsl; r++) {
                    idx0 = r * rowStridel;
                    for (long s = 0; s < slicesl; s++) {
                        idx1 = s * sliceStridel + idx0;
                        idx2 = 2 * s;
                        idx3 = 2 * slicesl + 2 * s;
                        t.setDouble(idx2, a.getDouble(idx1));
                        t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                        t.setDouble(idx3, a.getDouble(idx1 + 2));
                        t.setDouble(idx3 + 1, a.getDouble(idx1 + 3));
                    }
                    fftSlices.complexForward(t, 0);
                    fftSlices.complexForward(t, 2 * slicesl);
                    for (long s = 0; s < slicesl; s++) {
                        idx1 = s * sliceStridel + idx0;
                        idx2 = 2 * s;
                        idx3 = 2 * slicesl + 2 * s;
                        a.setDouble(idx1, t.getDouble(idx2));
                        a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                        a.setDouble(idx1 + 2, t.getDouble(idx3));
                        a.setDouble(idx1 + 3, t.getDouble(idx3 + 1));
                    }
                }
            } else if (columnsl == 2) {
                for (long r = 0; r < rowsl; r++) {
                    idx0 = r * rowStridel;
                    for (long s = 0; s < slicesl; s++) {
                        idx1 = s * sliceStridel + idx0;
                        idx2 = 2 * s;
                        t.setDouble(idx2, a.getDouble(idx1));
                        t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                    }
                    fftSlices.complexForward(t, 0);
                    for (long s = 0; s < slicesl; s++) {
                        idx1 = s * sliceStridel + idx0;
                        idx2 = 2 * s;
                        a.setDouble(idx1, t.getDouble(idx2));
                        a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                    }
                }
            }
        } else if (columnsl > 4) {
            for (long r = 0; r < rowsl; r++) {
                idx0 = r * rowStridel;
                for (long c = 0; c < columnsl; c += 8) {
                    for (long s = 0; s < slicesl; s++) {
                        idx1 = s * sliceStridel + idx0 + c;
                        idx2 = 2 * s;
                        idx3 = 2 * slicesl + 2 * s;
                        idx4 = idx3 + 2 * slicesl;
                        idx5 = idx4 + 2 * slicesl;
                        t.setDouble(idx2, a.getDouble(idx1));
                        t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                        t.setDouble(idx3, a.getDouble(idx1 + 2));
                        t.setDouble(idx3 + 1, a.getDouble(idx1 + 3));
                        t.setDouble(idx4, a.getDouble(idx1 + 4));
                        t.setDouble(idx4 + 1, a.getDouble(idx1 + 5));
                        t.setDouble(idx5, a.getDouble(idx1 + 6));
                        t.setDouble(idx5 + 1, a.getDouble(idx1 + 7));
                    }
                    fftSlices.complexInverse(t, 0, scale);
                    fftSlices.complexInverse(t, 2 * slicesl, scale);
                    fftSlices.complexInverse(t, 4 * slicesl, scale);
                    fftSlices.complexInverse(t, 6 * slicesl, scale);
                    for (long s = 0; s < slicesl; s++) {
                        idx1 = s * sliceStridel + idx0 + c;
                        idx2 = 2 * s;
                        idx3 = 2 * slicesl + 2 * s;
                        idx4 = idx3 + 2 * slicesl;
                        idx5 = idx4 + 2 * slicesl;
                        a.setDouble(idx1, t.getDouble(idx2));
                        a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                        a.setDouble(idx1 + 2, t.getDouble(idx3));
                        a.setDouble(idx1 + 3, t.getDouble(idx3 + 1));
                        a.setDouble(idx1 + 4, t.getDouble(idx4));
                        a.setDouble(idx1 + 5, t.getDouble(idx4 + 1));
                        a.setDouble(idx1 + 6, t.getDouble(idx5));
                        a.setDouble(idx1 + 7, t.getDouble(idx5 + 1));
                    }
                }
            }
        } else if (columnsl == 4) {
            for (long r = 0; r < rowsl; r++) {
                idx0 = r * rowStridel;
                for (long s = 0; s < slicesl; s++) {
                    idx1 = s * sliceStridel + idx0;
                    idx2 = 2 * s;
                    idx3 = 2 * slicesl + 2 * s;
                    t.setDouble(idx2, a.getDouble(idx1));
                    t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                    t.setDouble(idx3, a.getDouble(idx1 + 2));
                    t.setDouble(idx3 + 1, a.getDouble(idx1 + 3));
                }
                fftSlices.complexInverse(t, 0, scale);
                fftSlices.complexInverse(t, 2 * slicesl, scale);
                for (long s = 0; s < slicesl; s++) {
                    idx1 = s * sliceStridel + idx0;
                    idx2 = 2 * s;
                    idx3 = 2 * slicesl + 2 * s;
                    a.setDouble(idx1, t.getDouble(idx2));
                    a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                    a.setDouble(idx1 + 2, t.getDouble(idx3));
                    a.setDouble(idx1 + 3, t.getDouble(idx3 + 1));
                }
            }
        } else if (columnsl == 2) {
            for (long r = 0; r < rowsl; r++) {
                idx0 = r * rowStridel;
                for (long s = 0; s < slicesl; s++) {
                    idx1 = s * sliceStridel + idx0;
                    idx2 = 2 * s;
                    t.setDouble(idx2, a.getDouble(idx1));
                    t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                }
                fftSlices.complexInverse(t, 0, scale);
                for (long s = 0; s < slicesl; s++) {
                    idx1 = s * sliceStridel + idx0;
                    idx2 = 2 * s;
                    a.setDouble(idx1, t.getDouble(idx2));
                    a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                }
            }
        }
    }

    private void cdft3db_sub(int isgn, double[][][] a, boolean scale)
    {
        int idx2, idx3, idx4, idx5;
        int nt = slices;
        if (nt < rows) {
            nt = rows;
        }
        nt *= 8;
        if (columns == 4) {
            nt >>= 1;
        } else if (columns < 4) {
            nt >>= 2;
        }
        double[] t = new double[nt];
        if (isgn == -1) {
            if (columns > 4) {
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < columns; c += 8) {
                        for (int s = 0; s < slices; s++) {
                            idx2 = 2 * s;
                            idx3 = 2 * slices + 2 * s;
                            idx4 = idx3 + 2 * slices;
                            idx5 = idx4 + 2 * slices;
                            t[idx2] = a[s][r][c];
                            t[idx2 + 1] = a[s][r][c + 1];
                            t[idx3] = a[s][r][c + 2];
                            t[idx3 + 1] = a[s][r][c + 3];
                            t[idx4] = a[s][r][c + 4];
                            t[idx4 + 1] = a[s][r][c + 5];
                            t[idx5] = a[s][r][c + 6];
                            t[idx5 + 1] = a[s][r][c + 7];
                        }
                        fftSlices.complexForward(t, 0);
                        fftSlices.complexForward(t, 2 * slices);
                        fftSlices.complexForward(t, 4 * slices);
                        fftSlices.complexForward(t, 6 * slices);
                        for (int s = 0; s < slices; s++) {
                            idx2 = 2 * s;
                            idx3 = 2 * slices + 2 * s;
                            idx4 = idx3 + 2 * slices;
                            idx5 = idx4 + 2 * slices;
                            a[s][r][c] = t[idx2];
                            a[s][r][c + 1] = t[idx2 + 1];
                            a[s][r][c + 2] = t[idx3];
                            a[s][r][c + 3] = t[idx3 + 1];
                            a[s][r][c + 4] = t[idx4];
                            a[s][r][c + 5] = t[idx4 + 1];
                            a[s][r][c + 6] = t[idx5];
                            a[s][r][c + 7] = t[idx5 + 1];
                        }
                    }
                }
            } else if (columns == 4) {
                for (int r = 0; r < rows; r++) {
                    for (int s = 0; s < slices; s++) {
                        idx2 = 2 * s;
                        idx3 = 2 * slices + 2 * s;
                        t[idx2] = a[s][r][0];
                        t[idx2 + 1] = a[s][r][1];
                        t[idx3] = a[s][r][2];
                        t[idx3 + 1] = a[s][r][3];
                    }
                    fftSlices.complexForward(t, 0);
                    fftSlices.complexForward(t, 2 * slices);
                    for (int s = 0; s < slices; s++) {
                        idx2 = 2 * s;
                        idx3 = 2 * slices + 2 * s;
                        a[s][r][0] = t[idx2];
                        a[s][r][1] = t[idx2 + 1];
                        a[s][r][2] = t[idx3];
                        a[s][r][3] = t[idx3 + 1];
                    }
                }
            } else if (columns == 2) {
                for (int r = 0; r < rows; r++) {
                    for (int s = 0; s < slices; s++) {
                        idx2 = 2 * s;
                        t[idx2] = a[s][r][0];
                        t[idx2 + 1] = a[s][r][1];
                    }
                    fftSlices.complexForward(t, 0);
                    for (int s = 0; s < slices; s++) {
                        idx2 = 2 * s;
                        a[s][r][0] = t[idx2];
                        a[s][r][1] = t[idx2 + 1];
                    }
                }
            }
        } else if (columns > 4) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c += 8) {
                    for (int s = 0; s < slices; s++) {
                        idx2 = 2 * s;
                        idx3 = 2 * slices + 2 * s;
                        idx4 = idx3 + 2 * slices;
                        idx5 = idx4 + 2 * slices;
                        t[idx2] = a[s][r][c];
                        t[idx2 + 1] = a[s][r][c + 1];
                        t[idx3] = a[s][r][c + 2];
                        t[idx3 + 1] = a[s][r][c + 3];
                        t[idx4] = a[s][r][c + 4];
                        t[idx4 + 1] = a[s][r][c + 5];
                        t[idx5] = a[s][r][c + 6];
                        t[idx5 + 1] = a[s][r][c + 7];
                    }
                    fftSlices.complexInverse(t, 0, scale);
                    fftSlices.complexInverse(t, 2 * slices, scale);
                    fftSlices.complexInverse(t, 4 * slices, scale);
                    fftSlices.complexInverse(t, 6 * slices, scale);
                    for (int s = 0; s < slices; s++) {
                        idx2 = 2 * s;
                        idx3 = 2 * slices + 2 * s;
                        idx4 = idx3 + 2 * slices;
                        idx5 = idx4 + 2 * slices;
                        a[s][r][c] = t[idx2];
                        a[s][r][c + 1] = t[idx2 + 1];
                        a[s][r][c + 2] = t[idx3];
                        a[s][r][c + 3] = t[idx3 + 1];
                        a[s][r][c + 4] = t[idx4];
                        a[s][r][c + 5] = t[idx4 + 1];
                        a[s][r][c + 6] = t[idx5];
                        a[s][r][c + 7] = t[idx5 + 1];
                    }
                }
            }
        } else if (columns == 4) {
            for (int r = 0; r < rows; r++) {
                for (int s = 0; s < slices; s++) {
                    idx2 = 2 * s;
                    idx3 = 2 * slices + 2 * s;
                    t[idx2] = a[s][r][0];
                    t[idx2 + 1] = a[s][r][1];
                    t[idx3] = a[s][r][2];
                    t[idx3 + 1] = a[s][r][3];
                }
                fftSlices.complexInverse(t, 0, scale);
                fftSlices.complexInverse(t, 2 * slices, scale);
                for (int s = 0; s < slices; s++) {
                    idx2 = 2 * s;
                    idx3 = 2 * slices + 2 * s;
                    a[s][r][0] = t[idx2];
                    a[s][r][1] = t[idx2 + 1];
                    a[s][r][2] = t[idx3];
                    a[s][r][3] = t[idx3 + 1];
                }
            }
        } else if (columns == 2) {
            for (int r = 0; r < rows; r++) {
                for (int s = 0; s < slices; s++) {
                    idx2 = 2 * s;
                    t[idx2] = a[s][r][0];
                    t[idx2 + 1] = a[s][r][1];
                }
                fftSlices.complexInverse(t, 0, scale);
                for (int s = 0; s < slices; s++) {
                    idx2 = 2 * s;
                    a[s][r][0] = t[idx2];
                    a[s][r][1] = t[idx2 + 1];
                }
            }
        }
    }

    private void xdft3da_subth1(final int icr, final int isgn, final double[] a, final boolean scale)
    {
        int i;
        final int nthreads = min(ConcurrencyUtils.getNumberOfThreads(), slices);
        int nt = slices;
        if (nt < rows) {
            nt = rows;
        }
        nt *= 8;
        if (columns == 4) {
            nt >>= 1;
        } else if (columns < 4) {
            nt >>= 2;
        }
        final int ntf = nt;
        Future<?>[] futures = new Future[nthreads];
        for (i = 0; i < nthreads; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {
                public void run()
                {
                    int idx0, idx1, idx2, idx3, idx4, idx5;
                    double[] t = new double[ntf];
                    if (isgn == -1) {
                        for (int s = n0; s < slices; s += nthreads) {
                            idx0 = s * sliceStride;
                            if (icr == 0) {
                                for (int r = 0; r < rows; r++) {
                                    fftColumns.complexForward(a, idx0 + r * rowStride);
                                }
                            } else {
                                for (int r = 0; r < rows; r++) {
                                    fftColumns.realForward(a, idx0 + r * rowStride);
                                }
                            }
                            if (columns > 4) {
                                for (int c = 0; c < columns; c += 8) {
                                    for (int r = 0; r < rows; r++) {
                                        idx1 = idx0 + r * rowStride + c;
                                        idx2 = 2 * r;
                                        idx3 = 2 * rows + 2 * r;
                                        idx4 = idx3 + 2 * rows;
                                        idx5 = idx4 + 2 * rows;
                                        t[idx2] = a[idx1];
                                        t[idx2 + 1] = a[idx1 + 1];
                                        t[idx3] = a[idx1 + 2];
                                        t[idx3 + 1] = a[idx1 + 3];
                                        t[idx4] = a[idx1 + 4];
                                        t[idx4 + 1] = a[idx1 + 5];
                                        t[idx5] = a[idx1 + 6];
                                        t[idx5 + 1] = a[idx1 + 7];
                                    }
                                    fftRows.complexForward(t, 0);
                                    fftRows.complexForward(t, 2 * rows);
                                    fftRows.complexForward(t, 4 * rows);
                                    fftRows.complexForward(t, 6 * rows);
                                    for (int r = 0; r < rows; r++) {
                                        idx1 = idx0 + r * rowStride + c;
                                        idx2 = 2 * r;
                                        idx3 = 2 * rows + 2 * r;
                                        idx4 = idx3 + 2 * rows;
                                        idx5 = idx4 + 2 * rows;
                                        a[idx1] = t[idx2];
                                        a[idx1 + 1] = t[idx2 + 1];
                                        a[idx1 + 2] = t[idx3];
                                        a[idx1 + 3] = t[idx3 + 1];
                                        a[idx1 + 4] = t[idx4];
                                        a[idx1 + 5] = t[idx4 + 1];
                                        a[idx1 + 6] = t[idx5];
                                        a[idx1 + 7] = t[idx5 + 1];
                                    }
                                }
                            } else if (columns == 4) {
                                for (int r = 0; r < rows; r++) {
                                    idx1 = idx0 + r * rowStride;
                                    idx2 = 2 * r;
                                    idx3 = 2 * rows + 2 * r;
                                    t[idx2] = a[idx1];
                                    t[idx2 + 1] = a[idx1 + 1];
                                    t[idx3] = a[idx1 + 2];
                                    t[idx3 + 1] = a[idx1 + 3];
                                }
                                fftRows.complexForward(t, 0);
                                fftRows.complexForward(t, 2 * rows);
                                for (int r = 0; r < rows; r++) {
                                    idx1 = idx0 + r * rowStride;
                                    idx2 = 2 * r;
                                    idx3 = 2 * rows + 2 * r;
                                    a[idx1] = t[idx2];
                                    a[idx1 + 1] = t[idx2 + 1];
                                    a[idx1 + 2] = t[idx3];
                                    a[idx1 + 3] = t[idx3 + 1];
                                }
                            } else if (columns == 2) {
                                for (int r = 0; r < rows; r++) {
                                    idx1 = idx0 + r * rowStride;
                                    idx2 = 2 * r;
                                    t[idx2] = a[idx1];
                                    t[idx2 + 1] = a[idx1 + 1];
                                }
                                fftRows.complexForward(t, 0);
                                for (int r = 0; r < rows; r++) {
                                    idx1 = idx0 + r * rowStride;
                                    idx2 = 2 * r;
                                    a[idx1] = t[idx2];
                                    a[idx1 + 1] = t[idx2 + 1];
                                }
                            }

                        }
                    } else {
                        for (int s = n0; s < slices; s += nthreads) {
                            idx0 = s * sliceStride;
                            if (icr == 0) {
                                for (int r = 0; r < rows; r++) {
                                    fftColumns.complexInverse(a, idx0 + r * rowStride, scale);
                                }
                            }
                            if (columns > 4) {
                                for (int c = 0; c < columns; c += 8) {
                                    for (int r = 0; r < rows; r++) {
                                        idx1 = idx0 + r * rowStride + c;
                                        idx2 = 2 * r;
                                        idx3 = 2 * rows + 2 * r;
                                        idx4 = idx3 + 2 * rows;
                                        idx5 = idx4 + 2 * rows;
                                        t[idx2] = a[idx1];
                                        t[idx2 + 1] = a[idx1 + 1];
                                        t[idx3] = a[idx1 + 2];
                                        t[idx3 + 1] = a[idx1 + 3];
                                        t[idx4] = a[idx1 + 4];
                                        t[idx4 + 1] = a[idx1 + 5];
                                        t[idx5] = a[idx1 + 6];
                                        t[idx5 + 1] = a[idx1 + 7];
                                    }
                                    fftRows.complexInverse(t, 0, scale);
                                    fftRows.complexInverse(t, 2 * rows, scale);
                                    fftRows.complexInverse(t, 4 * rows, scale);
                                    fftRows.complexInverse(t, 6 * rows, scale);
                                    for (int r = 0; r < rows; r++) {
                                        idx1 = idx0 + r * rowStride + c;
                                        idx2 = 2 * r;
                                        idx3 = 2 * rows + 2 * r;
                                        idx4 = idx3 + 2 * rows;
                                        idx5 = idx4 + 2 * rows;
                                        a[idx1] = t[idx2];
                                        a[idx1 + 1] = t[idx2 + 1];
                                        a[idx1 + 2] = t[idx3];
                                        a[idx1 + 3] = t[idx3 + 1];
                                        a[idx1 + 4] = t[idx4];
                                        a[idx1 + 5] = t[idx4 + 1];
                                        a[idx1 + 6] = t[idx5];
                                        a[idx1 + 7] = t[idx5 + 1];
                                    }
                                }
                            } else if (columns == 4) {
                                for (int r = 0; r < rows; r++) {
                                    idx1 = idx0 + r * rowStride;
                                    idx2 = 2 * r;
                                    idx3 = 2 * rows + 2 * r;
                                    t[idx2] = a[idx1];
                                    t[idx2 + 1] = a[idx1 + 1];
                                    t[idx3] = a[idx1 + 2];
                                    t[idx3 + 1] = a[idx1 + 3];
                                }
                                fftRows.complexInverse(t, 0, scale);
                                fftRows.complexInverse(t, 2 * rows, scale);
                                for (int r = 0; r < rows; r++) {
                                    idx1 = idx0 + r * rowStride;
                                    idx2 = 2 * r;
                                    idx3 = 2 * rows + 2 * r;
                                    a[idx1] = t[idx2];
                                    a[idx1 + 1] = t[idx2 + 1];
                                    a[idx1 + 2] = t[idx3];
                                    a[idx1 + 3] = t[idx3 + 1];
                                }
                            } else if (columns == 2) {
                                for (int r = 0; r < rows; r++) {
                                    idx1 = idx0 + r * rowStride;
                                    idx2 = 2 * r;
                                    t[idx2] = a[idx1];
                                    t[idx2 + 1] = a[idx1 + 1];
                                }
                                fftRows.complexInverse(t, 0, scale);
                                for (int r = 0; r < rows; r++) {
                                    idx1 = idx0 + r * rowStride;
                                    idx2 = 2 * r;
                                    a[idx1] = t[idx2];
                                    a[idx1 + 1] = t[idx2 + 1];
                                }
                            }
                            if (icr != 0) {
                                for (int r = 0; r < rows; r++) {
                                    fftColumns.realInverse(a, idx0 + r * rowStride, scale);
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
            Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void xdft3da_subth1(final long icr, final int isgn, final DoubleLargeArray a, final boolean scale)
    {
        int i;
        final int nthreads = (int) min(ConcurrencyUtils.getNumberOfThreads(), slicesl);
        long nt = slicesl;
        if (nt < rowsl) {
            nt = rowsl;
        }
        nt *= 8;
        if (columnsl == 4) {
            nt >>= 1;
        } else if (columnsl < 4) {
            nt >>= 2;
        }
        final long ntf = nt;
        Future<?>[] futures = new Future[nthreads];
        for (i = 0; i < nthreads; i++) {
            final long n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {
                public void run()
                {
                    long idx0, idx1, idx2, idx3, idx4, idx5;
                    DoubleLargeArray t = new DoubleLargeArray(ntf);
                    if (isgn == -1) {
                        for (long s = n0; s < slicesl; s += nthreads) {
                            idx0 = s * sliceStridel;
                            if (icr == 0) {
                                for (long r = 0; r < rowsl; r++) {
                                    fftColumns.complexForward(a, idx0 + r * rowStridel);
                                }
                            } else {
                                for (long r = 0; r < rowsl; r++) {
                                    fftColumns.realForward(a, idx0 + r * rowStridel);
                                }
                            }
                            if (columnsl > 4) {
                                for (long c = 0; c < columnsl; c += 8) {
                                    for (long r = 0; r < rowsl; r++) {
                                        idx1 = idx0 + r * rowStridel + c;
                                        idx2 = 2 * r;
                                        idx3 = 2 * rowsl + 2 * r;
                                        idx4 = idx3 + 2 * rowsl;
                                        idx5 = idx4 + 2 * rowsl;
                                        t.setDouble(idx2, a.getDouble(idx1));
                                        t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                                        t.setDouble(idx3, a.getDouble(idx1 + 2));
                                        t.setDouble(idx3 + 1, a.getDouble(idx1 + 3));
                                        t.setDouble(idx4, a.getDouble(idx1 + 4));
                                        t.setDouble(idx4 + 1, a.getDouble(idx1 + 5));
                                        t.setDouble(idx5, a.getDouble(idx1 + 6));
                                        t.setDouble(idx5 + 1, a.getDouble(idx1 + 7));
                                    }
                                    fftRows.complexForward(t, 0);
                                    fftRows.complexForward(t, 2 * rowsl);
                                    fftRows.complexForward(t, 4 * rowsl);
                                    fftRows.complexForward(t, 6 * rowsl);
                                    for (long r = 0; r < rowsl; r++) {
                                        idx1 = idx0 + r * rowStridel + c;
                                        idx2 = 2 * r;
                                        idx3 = 2 * rowsl + 2 * r;
                                        idx4 = idx3 + 2 * rowsl;
                                        idx5 = idx4 + 2 * rowsl;
                                        a.setDouble(idx1, t.getDouble(idx2));
                                        a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                                        a.setDouble(idx1 + 2, t.getDouble(idx3));
                                        a.setDouble(idx1 + 3, t.getDouble(idx3 + 1));
                                        a.setDouble(idx1 + 4, t.getDouble(idx4));
                                        a.setDouble(idx1 + 5, t.getDouble(idx4 + 1));
                                        a.setDouble(idx1 + 6, t.getDouble(idx5));
                                        a.setDouble(idx1 + 7, t.getDouble(idx5 + 1));
                                    }
                                }
                            } else if (columnsl == 4) {
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStridel;
                                    idx2 = 2 * r;
                                    idx3 = 2 * rowsl + 2 * r;
                                    t.setDouble(idx2, a.getDouble(idx1));
                                    t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                                    t.setDouble(idx3, a.getDouble(idx1 + 2));
                                    t.setDouble(idx3 + 1, a.getDouble(idx1 + 3));
                                }
                                fftRows.complexForward(t, 0);
                                fftRows.complexForward(t, 2 * rowsl);
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStridel;
                                    idx2 = 2 * r;
                                    idx3 = 2 * rowsl + 2 * r;
                                    a.setDouble(idx1, t.getDouble(idx2));
                                    a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                                    a.setDouble(idx1 + 2, t.getDouble(idx3));
                                    a.setDouble(idx1 + 3, t.getDouble(idx3 + 1));
                                }
                            } else if (columnsl == 2) {
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStridel;
                                    idx2 = 2 * r;
                                    t.setDouble(idx2, a.getDouble(idx1));
                                    t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                                }
                                fftRows.complexForward(t, 0);
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStridel;
                                    idx2 = 2 * r;
                                    a.setDouble(idx1, t.getDouble(idx2));
                                    a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                                }
                            }

                        }
                    } else {
                        for (long s = n0; s < slicesl; s += nthreads) {
                            idx0 = s * sliceStridel;
                            if (icr == 0) {
                                for (long r = 0; r < rowsl; r++) {
                                    fftColumns.complexInverse(a, idx0 + r * rowStridel, scale);
                                }
                            }
                            if (columnsl > 4) {
                                for (long c = 0; c < columnsl; c += 8) {
                                    for (long r = 0; r < rowsl; r++) {
                                        idx1 = idx0 + r * rowStridel + c;
                                        idx2 = 2 * r;
                                        idx3 = 2 * rowsl + 2 * r;
                                        idx4 = idx3 + 2 * rowsl;
                                        idx5 = idx4 + 2 * rowsl;
                                        t.setDouble(idx2, a.getDouble(idx1));
                                        t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                                        t.setDouble(idx3, a.getDouble(idx1 + 2));
                                        t.setDouble(idx3 + 1, a.getDouble(idx1 + 3));
                                        t.setDouble(idx4, a.getDouble(idx1 + 4));
                                        t.setDouble(idx4 + 1, a.getDouble(idx1 + 5));
                                        t.setDouble(idx5, a.getDouble(idx1 + 6));
                                        t.setDouble(idx5 + 1, a.getDouble(idx1 + 7));
                                    }
                                    fftRows.complexInverse(t, 0, scale);
                                    fftRows.complexInverse(t, 2 * rowsl, scale);
                                    fftRows.complexInverse(t, 4 * rowsl, scale);
                                    fftRows.complexInverse(t, 6 * rowsl, scale);
                                    for (long r = 0; r < rowsl; r++) {
                                        idx1 = idx0 + r * rowStridel + c;
                                        idx2 = 2 * r;
                                        idx3 = 2 * rowsl + 2 * r;
                                        idx4 = idx3 + 2 * rowsl;
                                        idx5 = idx4 + 2 * rowsl;
                                        a.setDouble(idx1, t.getDouble(idx2));
                                        a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                                        a.setDouble(idx1 + 2, t.getDouble(idx3));
                                        a.setDouble(idx1 + 3, t.getDouble(idx3 + 1));
                                        a.setDouble(idx1 + 4, t.getDouble(idx4));
                                        a.setDouble(idx1 + 5, t.getDouble(idx4 + 1));
                                        a.setDouble(idx1 + 6, t.getDouble(idx5));
                                        a.setDouble(idx1 + 7, t.getDouble(idx5 + 1));
                                    }
                                }
                            } else if (columnsl == 4) {
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStridel;
                                    idx2 = 2 * r;
                                    idx3 = 2 * rowsl + 2 * r;
                                    t.setDouble(idx2, a.getDouble(idx1));
                                    t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                                    t.setDouble(idx3, a.getDouble(idx1 + 2));
                                    t.setDouble(idx3 + 1, a.getDouble(idx1 + 3));
                                }
                                fftRows.complexInverse(t, 0, scale);
                                fftRows.complexInverse(t, 2 * rowsl, scale);
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStridel;
                                    idx2 = 2 * r;
                                    idx3 = 2 * rowsl + 2 * r;
                                    a.setDouble(idx1, t.getDouble(idx2));
                                    a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                                    a.setDouble(idx1 + 2, t.getDouble(idx3));
                                    a.setDouble(idx1 + 3, t.getDouble(idx3 + 1));
                                }
                            } else if (columnsl == 2) {
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStridel;
                                    idx2 = 2 * r;
                                    t.setDouble(idx2, a.getDouble(idx1));
                                    t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                                }
                                fftRows.complexInverse(t, 0, scale);
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStridel;
                                    idx2 = 2 * r;
                                    a.setDouble(idx1, t.getDouble(idx2));
                                    a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                                }
                            }
                            if (icr != 0) {
                                for (long r = 0; r < rowsl; r++) {
                                    fftColumns.realInverse(a, idx0 + r * rowStridel, scale);
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
            Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void xdft3da_subth2(final int icr, final int isgn, final double[] a, final boolean scale)
    {
        int i;
        final int nthreads = min(ConcurrencyUtils.getNumberOfThreads(), slices);
        int nt = slices;
        if (nt < rows) {
            nt = rows;
        }
        nt *= 8;
        if (columns == 4) {
            nt >>= 1;
        } else if (columns < 4) {
            nt >>= 2;
        }
        final int ntf = nt;
        Future<?>[] futures = new Future[nthreads];
        for (i = 0; i < nthreads; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {
                public void run()
                {
                    int idx0, idx1, idx2, idx3, idx4, idx5;
                    double[] t = new double[ntf];
                    if (isgn == -1) {
                        for (int s = n0; s < slices; s += nthreads) {
                            idx0 = s * sliceStride;
                            if (icr == 0) {
                                for (int r = 0; r < rows; r++) {
                                    fftColumns.complexForward(a, idx0 + r * rowStride);
                                }
                            } else {
                                for (int r = 0; r < rows; r++) {
                                    fftColumns.realForward(a, idx0 + r * rowStride);
                                }
                            }
                            if (columns > 4) {
                                for (int c = 0; c < columns; c += 8) {
                                    for (int r = 0; r < rows; r++) {
                                        idx1 = idx0 + r * rowStride + c;
                                        idx2 = 2 * r;
                                        idx3 = 2 * rows + 2 * r;
                                        idx4 = idx3 + 2 * rows;
                                        idx5 = idx4 + 2 * rows;
                                        t[idx2] = a[idx1];
                                        t[idx2 + 1] = a[idx1 + 1];
                                        t[idx3] = a[idx1 + 2];
                                        t[idx3 + 1] = a[idx1 + 3];
                                        t[idx4] = a[idx1 + 4];
                                        t[idx4 + 1] = a[idx1 + 5];
                                        t[idx5] = a[idx1 + 6];
                                        t[idx5 + 1] = a[idx1 + 7];
                                    }
                                    fftRows.complexForward(t, 0);
                                    fftRows.complexForward(t, 2 * rows);
                                    fftRows.complexForward(t, 4 * rows);
                                    fftRows.complexForward(t, 6 * rows);
                                    for (int r = 0; r < rows; r++) {
                                        idx1 = idx0 + r * rowStride + c;
                                        idx2 = 2 * r;
                                        idx3 = 2 * rows + 2 * r;
                                        idx4 = idx3 + 2 * rows;
                                        idx5 = idx4 + 2 * rows;
                                        a[idx1] = t[idx2];
                                        a[idx1 + 1] = t[idx2 + 1];
                                        a[idx1 + 2] = t[idx3];
                                        a[idx1 + 3] = t[idx3 + 1];
                                        a[idx1 + 4] = t[idx4];
                                        a[idx1 + 5] = t[idx4 + 1];
                                        a[idx1 + 6] = t[idx5];
                                        a[idx1 + 7] = t[idx5 + 1];
                                    }
                                }
                            } else if (columns == 4) {
                                for (int r = 0; r < rows; r++) {
                                    idx1 = idx0 + r * rowStride;
                                    idx2 = 2 * r;
                                    idx3 = 2 * rows + 2 * r;
                                    t[idx2] = a[idx1];
                                    t[idx2 + 1] = a[idx1 + 1];
                                    t[idx3] = a[idx1 + 2];
                                    t[idx3 + 1] = a[idx1 + 3];
                                }
                                fftRows.complexForward(t, 0);
                                fftRows.complexForward(t, 2 * rows);
                                for (int r = 0; r < rows; r++) {
                                    idx1 = idx0 + r * rowStride;
                                    idx2 = 2 * r;
                                    idx3 = 2 * rows + 2 * r;
                                    a[idx1] = t[idx2];
                                    a[idx1 + 1] = t[idx2 + 1];
                                    a[idx1 + 2] = t[idx3];
                                    a[idx1 + 3] = t[idx3 + 1];
                                }
                            } else if (columns == 2) {
                                for (int r = 0; r < rows; r++) {
                                    idx1 = idx0 + r * rowStride;
                                    idx2 = 2 * r;
                                    t[idx2] = a[idx1];
                                    t[idx2 + 1] = a[idx1 + 1];
                                }
                                fftRows.complexForward(t, 0);
                                for (int r = 0; r < rows; r++) {
                                    idx1 = idx0 + r * rowStride;
                                    idx2 = 2 * r;
                                    a[idx1] = t[idx2];
                                    a[idx1 + 1] = t[idx2 + 1];
                                }
                            }

                        }
                    } else {
                        for (int s = n0; s < slices; s += nthreads) {
                            idx0 = s * sliceStride;
                            if (icr == 0) {
                                for (int r = 0; r < rows; r++) {
                                    fftColumns.complexInverse(a, idx0 + r * rowStride, scale);
                                }
                            } else {
                                for (int r = 0; r < rows; r++) {
                                    fftColumns.realInverse2(a, idx0 + r * rowStride, scale);
                                }
                            }
                            if (columns > 4) {
                                for (int c = 0; c < columns; c += 8) {
                                    for (int r = 0; r < rows; r++) {
                                        idx1 = idx0 + r * rowStride + c;
                                        idx2 = 2 * r;
                                        idx3 = 2 * rows + 2 * r;
                                        idx4 = idx3 + 2 * rows;
                                        idx5 = idx4 + 2 * rows;
                                        t[idx2] = a[idx1];
                                        t[idx2 + 1] = a[idx1 + 1];
                                        t[idx3] = a[idx1 + 2];
                                        t[idx3 + 1] = a[idx1 + 3];
                                        t[idx4] = a[idx1 + 4];
                                        t[idx4 + 1] = a[idx1 + 5];
                                        t[idx5] = a[idx1 + 6];
                                        t[idx5 + 1] = a[idx1 + 7];
                                    }
                                    fftRows.complexInverse(t, 0, scale);
                                    fftRows.complexInverse(t, 2 * rows, scale);
                                    fftRows.complexInverse(t, 4 * rows, scale);
                                    fftRows.complexInverse(t, 6 * rows, scale);
                                    for (int r = 0; r < rows; r++) {
                                        idx1 = idx0 + r * rowStride + c;
                                        idx2 = 2 * r;
                                        idx3 = 2 * rows + 2 * r;
                                        idx4 = idx3 + 2 * rows;
                                        idx5 = idx4 + 2 * rows;
                                        a[idx1] = t[idx2];
                                        a[idx1 + 1] = t[idx2 + 1];
                                        a[idx1 + 2] = t[idx3];
                                        a[idx1 + 3] = t[idx3 + 1];
                                        a[idx1 + 4] = t[idx4];
                                        a[idx1 + 5] = t[idx4 + 1];
                                        a[idx1 + 6] = t[idx5];
                                        a[idx1 + 7] = t[idx5 + 1];
                                    }
                                }
                            } else if (columns == 4) {
                                for (int r = 0; r < rows; r++) {
                                    idx1 = idx0 + r * rowStride;
                                    idx2 = 2 * r;
                                    idx3 = 2 * rows + 2 * r;
                                    t[idx2] = a[idx1];
                                    t[idx2 + 1] = a[idx1 + 1];
                                    t[idx3] = a[idx1 + 2];
                                    t[idx3 + 1] = a[idx1 + 3];
                                }
                                fftRows.complexInverse(t, 0, scale);
                                fftRows.complexInverse(t, 2 * rows, scale);
                                for (int r = 0; r < rows; r++) {
                                    idx1 = idx0 + r * rowStride;
                                    idx2 = 2 * r;
                                    idx3 = 2 * rows + 2 * r;
                                    a[idx1] = t[idx2];
                                    a[idx1 + 1] = t[idx2 + 1];
                                    a[idx1 + 2] = t[idx3];
                                    a[idx1 + 3] = t[idx3 + 1];
                                }
                            } else if (columns == 2) {
                                for (int r = 0; r < rows; r++) {
                                    idx1 = idx0 + r * rowStride;
                                    idx2 = 2 * r;
                                    t[idx2] = a[idx1];
                                    t[idx2 + 1] = a[idx1 + 1];
                                }
                                fftRows.complexInverse(t, 0, scale);
                                for (int r = 0; r < rows; r++) {
                                    idx1 = idx0 + r * rowStride;
                                    idx2 = 2 * r;
                                    a[idx1] = t[idx2];
                                    a[idx1 + 1] = t[idx2 + 1];
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
            Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void xdft3da_subth2(final long icr, final int isgn, final DoubleLargeArray a, final boolean scale)
    {
        int i;
        final int nthreads = (int) min(ConcurrencyUtils.getNumberOfThreads(), slicesl);
        long nt = slicesl;
        if (nt < rowsl) {
            nt = rowsl;
        }
        nt *= 8;
        if (columnsl == 4) {
            nt >>= 1;
        } else if (columnsl < 4) {
            nt >>= 2;
        }
        final long ntf = nt;
        Future<?>[] futures = new Future[nthreads];
        for (i = 0; i < nthreads; i++) {
            final long n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {
                public void run()
                {
                    long idx0, idx1, idx2, idx3, idx4, idx5;
                    DoubleLargeArray t = new DoubleLargeArray(ntf);
                    if (isgn == -1) {
                        for (long s = n0; s < slicesl; s += nthreads) {
                            idx0 = s * sliceStridel;
                            if (icr == 0) {
                                for (long r = 0; r < rowsl; r++) {
                                    fftColumns.complexForward(a, idx0 + r * rowStridel);
                                }
                            } else {
                                for (long r = 0; r < rowsl; r++) {
                                    fftColumns.realForward(a, idx0 + r * rowStridel);
                                }
                            }
                            if (columnsl > 4) {
                                for (long c = 0; c < columnsl; c += 8) {
                                    for (long r = 0; r < rowsl; r++) {
                                        idx1 = idx0 + r * rowStridel + c;
                                        idx2 = 2 * r;
                                        idx3 = 2 * rowsl + 2 * r;
                                        idx4 = idx3 + 2 * rowsl;
                                        idx5 = idx4 + 2 * rowsl;
                                        t.setDouble(idx2, a.getDouble(idx1));
                                        t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                                        t.setDouble(idx3, a.getDouble(idx1 + 2));
                                        t.setDouble(idx3 + 1, a.getDouble(idx1 + 3));
                                        t.setDouble(idx4, a.getDouble(idx1 + 4));
                                        t.setDouble(idx4 + 1, a.getDouble(idx1 + 5));
                                        t.setDouble(idx5, a.getDouble(idx1 + 6));
                                        t.setDouble(idx5 + 1, a.getDouble(idx1 + 7));
                                    }
                                    fftRows.complexForward(t, 0);
                                    fftRows.complexForward(t, 2 * rowsl);
                                    fftRows.complexForward(t, 4 * rowsl);
                                    fftRows.complexForward(t, 6 * rowsl);
                                    for (long r = 0; r < rowsl; r++) {
                                        idx1 = idx0 + r * rowStridel + c;
                                        idx2 = 2 * r;
                                        idx3 = 2 * rowsl + 2 * r;
                                        idx4 = idx3 + 2 * rowsl;
                                        idx5 = idx4 + 2 * rowsl;
                                        a.setDouble(idx1, t.getDouble(idx2));
                                        a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                                        a.setDouble(idx1 + 2, t.getDouble(idx3));
                                        a.setDouble(idx1 + 3, t.getDouble(idx3 + 1));
                                        a.setDouble(idx1 + 4, t.getDouble(idx4));
                                        a.setDouble(idx1 + 5, t.getDouble(idx4 + 1));
                                        a.setDouble(idx1 + 6, t.getDouble(idx5));
                                        a.setDouble(idx1 + 7, t.getDouble(idx5 + 1));
                                    }
                                }
                            } else if (columnsl == 4) {
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStridel;
                                    idx2 = 2 * r;
                                    idx3 = 2 * rowsl + 2 * r;
                                    t.setDouble(idx2, a.getDouble(idx1));
                                    t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                                    t.setDouble(idx3, a.getDouble(idx1 + 2));
                                    t.setDouble(idx3 + 1, a.getDouble(idx1 + 3));
                                }
                                fftRows.complexForward(t, 0);
                                fftRows.complexForward(t, 2 * rowsl);
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStridel;
                                    idx2 = 2 * r;
                                    idx3 = 2 * rowsl + 2 * r;
                                    a.setDouble(idx1, t.getDouble(idx2));
                                    a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                                    a.setDouble(idx1 + 2, t.getDouble(idx3));
                                    a.setDouble(idx1 + 3, t.getDouble(idx3 + 1));
                                }
                            } else if (columnsl == 2) {
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStridel;
                                    idx2 = 2 * r;
                                    t.setDouble(idx2, a.getDouble(idx1));
                                    t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                                }
                                fftRows.complexForward(t, 0);
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStridel;
                                    idx2 = 2 * r;
                                    a.setDouble(idx1, t.getDouble(idx2));
                                    a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                                }
                            }

                        }
                    } else {
                        for (long s = n0; s < slicesl; s += nthreads) {
                            idx0 = s * sliceStridel;
                            if (icr == 0) {
                                for (long r = 0; r < rowsl; r++) {
                                    fftColumns.complexInverse(a, idx0 + r * rowStridel, scale);
                                }
                            } else {
                                for (long r = 0; r < rowsl; r++) {
                                    fftColumns.realInverse2(a, idx0 + r * rowStridel, scale);
                                }
                            }
                            if (columnsl > 4) {
                                for (long c = 0; c < columnsl; c += 8) {
                                    for (long r = 0; r < rowsl; r++) {
                                        idx1 = idx0 + r * rowStridel + c;
                                        idx2 = 2 * r;
                                        idx3 = 2 * rowsl + 2 * r;
                                        idx4 = idx3 + 2 * rowsl;
                                        idx5 = idx4 + 2 * rowsl;
                                        t.setDouble(idx2, a.getDouble(idx1));
                                        t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                                        t.setDouble(idx3, a.getDouble(idx1 + 2));
                                        t.setDouble(idx3 + 1, a.getDouble(idx1 + 3));
                                        t.setDouble(idx4, a.getDouble(idx1 + 4));
                                        t.setDouble(idx4 + 1, a.getDouble(idx1 + 5));
                                        t.setDouble(idx5, a.getDouble(idx1 + 6));
                                        t.setDouble(idx5 + 1, a.getDouble(idx1 + 7));
                                    }
                                    fftRows.complexInverse(t, 0, scale);
                                    fftRows.complexInverse(t, 2 * rowsl, scale);
                                    fftRows.complexInverse(t, 4 * rowsl, scale);
                                    fftRows.complexInverse(t, 6 * rowsl, scale);
                                    for (long r = 0; r < rowsl; r++) {
                                        idx1 = idx0 + r * rowStridel + c;
                                        idx2 = 2 * r;
                                        idx3 = 2 * rowsl + 2 * r;
                                        idx4 = idx3 + 2 * rowsl;
                                        idx5 = idx4 + 2 * rowsl;
                                        a.setDouble(idx1, t.getDouble(idx2));
                                        a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                                        a.setDouble(idx1 + 2, t.getDouble(idx3));
                                        a.setDouble(idx1 + 3, t.getDouble(idx3 + 1));
                                        a.setDouble(idx1 + 4, t.getDouble(idx4));
                                        a.setDouble(idx1 + 5, t.getDouble(idx4 + 1));
                                        a.setDouble(idx1 + 6, t.getDouble(idx5));
                                        a.setDouble(idx1 + 7, t.getDouble(idx5 + 1));
                                    }
                                }
                            } else if (columnsl == 4) {
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStridel;
                                    idx2 = 2 * r;
                                    idx3 = 2 * rowsl + 2 * r;
                                    t.setDouble(idx2, a.getDouble(idx1));
                                    t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                                    t.setDouble(idx3, a.getDouble(idx1 + 2));
                                    t.setDouble(idx3 + 1, a.getDouble(idx1 + 3));
                                }
                                fftRows.complexInverse(t, 0, scale);
                                fftRows.complexInverse(t, 2 * rowsl, scale);
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStridel;
                                    idx2 = 2 * r;
                                    idx3 = 2 * rowsl + 2 * r;
                                    a.setDouble(idx1, t.getDouble(idx2));
                                    a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                                    a.setDouble(idx1 + 2, t.getDouble(idx3));
                                    a.setDouble(idx1 + 3, t.getDouble(idx3 + 1));
                                }
                            } else if (columnsl == 2) {
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStridel;
                                    idx2 = 2 * r;
                                    t.setDouble(idx2, a.getDouble(idx1));
                                    t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                                }
                                fftRows.complexInverse(t, 0, scale);
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = idx0 + r * rowStridel;
                                    idx2 = 2 * r;
                                    a.setDouble(idx1, t.getDouble(idx2));
                                    a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
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
            Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void xdft3da_subth1(final int icr, final int isgn, final double[][][] a, final boolean scale)
    {
        int i;
        final int nthreads = min(ConcurrencyUtils.getNumberOfThreads(), slices);
        int nt = slices;
        if (nt < rows) {
            nt = rows;
        }
        nt *= 8;
        if (columns == 4) {
            nt >>= 1;
        } else if (columns < 4) {
            nt >>= 2;
        }
        final int ntf = nt;
        Future<?>[] futures = new Future[nthreads];
        for (i = 0; i < nthreads; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {
                public void run()
                {
                    int idx2, idx3, idx4, idx5;
                    double[] t = new double[ntf];
                    if (isgn == -1) {
                        for (int s = n0; s < slices; s += nthreads) {
                            if (icr == 0) {
                                for (int r = 0; r < rows; r++) {
                                    fftColumns.complexForward(a[s][r]);
                                }
                            } else {
                                for (int r = 0; r < rows; r++) {
                                    fftColumns.realForward(a[s][r], 0);
                                }
                            }
                            if (columns > 4) {
                                for (int c = 0; c < columns; c += 8) {
                                    for (int r = 0; r < rows; r++) {
                                        idx2 = 2 * r;
                                        idx3 = 2 * rows + 2 * r;
                                        idx4 = idx3 + 2 * rows;
                                        idx5 = idx4 + 2 * rows;
                                        t[idx2] = a[s][r][c];
                                        t[idx2 + 1] = a[s][r][c + 1];
                                        t[idx3] = a[s][r][c + 2];
                                        t[idx3 + 1] = a[s][r][c + 3];
                                        t[idx4] = a[s][r][c + 4];
                                        t[idx4 + 1] = a[s][r][c + 5];
                                        t[idx5] = a[s][r][c + 6];
                                        t[idx5 + 1] = a[s][r][c + 7];
                                    }
                                    fftRows.complexForward(t, 0);
                                    fftRows.complexForward(t, 2 * rows);
                                    fftRows.complexForward(t, 4 * rows);
                                    fftRows.complexForward(t, 6 * rows);
                                    for (int r = 0; r < rows; r++) {
                                        idx2 = 2 * r;
                                        idx3 = 2 * rows + 2 * r;
                                        idx4 = idx3 + 2 * rows;
                                        idx5 = idx4 + 2 * rows;
                                        a[s][r][c] = t[idx2];
                                        a[s][r][c + 1] = t[idx2 + 1];
                                        a[s][r][c + 2] = t[idx3];
                                        a[s][r][c + 3] = t[idx3 + 1];
                                        a[s][r][c + 4] = t[idx4];
                                        a[s][r][c + 5] = t[idx4 + 1];
                                        a[s][r][c + 6] = t[idx5];
                                        a[s][r][c + 7] = t[idx5 + 1];
                                    }
                                }
                            } else if (columns == 4) {
                                for (int r = 0; r < rows; r++) {
                                    idx2 = 2 * r;
                                    idx3 = 2 * rows + 2 * r;
                                    t[idx2] = a[s][r][0];
                                    t[idx2 + 1] = a[s][r][1];
                                    t[idx3] = a[s][r][2];
                                    t[idx3 + 1] = a[s][r][3];
                                }
                                fftRows.complexForward(t, 0);
                                fftRows.complexForward(t, 2 * rows);
                                for (int r = 0; r < rows; r++) {
                                    idx2 = 2 * r;
                                    idx3 = 2 * rows + 2 * r;
                                    a[s][r][0] = t[idx2];
                                    a[s][r][1] = t[idx2 + 1];
                                    a[s][r][2] = t[idx3];
                                    a[s][r][3] = t[idx3 + 1];
                                }
                            } else if (columns == 2) {
                                for (int r = 0; r < rows; r++) {
                                    idx2 = 2 * r;
                                    t[idx2] = a[s][r][0];
                                    t[idx2 + 1] = a[s][r][1];
                                }
                                fftRows.complexForward(t, 0);
                                for (int r = 0; r < rows; r++) {
                                    idx2 = 2 * r;
                                    a[s][r][0] = t[idx2];
                                    a[s][r][1] = t[idx2 + 1];
                                }
                            }

                        }
                    } else {
                        for (int s = n0; s < slices; s += nthreads) {
                            if (icr == 0) {
                                for (int r = 0; r < rows; r++) {
                                    fftColumns.complexInverse(a[s][r], scale);
                                }
                            }
                            if (columns > 4) {
                                for (int c = 0; c < columns; c += 8) {
                                    for (int r = 0; r < rows; r++) {
                                        idx2 = 2 * r;
                                        idx3 = 2 * rows + 2 * r;
                                        idx4 = idx3 + 2 * rows;
                                        idx5 = idx4 + 2 * rows;
                                        t[idx2] = a[s][r][c];
                                        t[idx2 + 1] = a[s][r][c + 1];
                                        t[idx3] = a[s][r][c + 2];
                                        t[idx3 + 1] = a[s][r][c + 3];
                                        t[idx4] = a[s][r][c + 4];
                                        t[idx4 + 1] = a[s][r][c + 5];
                                        t[idx5] = a[s][r][c + 6];
                                        t[idx5 + 1] = a[s][r][c + 7];
                                    }
                                    fftRows.complexInverse(t, 0, scale);
                                    fftRows.complexInverse(t, 2 * rows, scale);
                                    fftRows.complexInverse(t, 4 * rows, scale);
                                    fftRows.complexInverse(t, 6 * rows, scale);
                                    for (int r = 0; r < rows; r++) {
                                        idx2 = 2 * r;
                                        idx3 = 2 * rows + 2 * r;
                                        idx4 = idx3 + 2 * rows;
                                        idx5 = idx4 + 2 * rows;
                                        a[s][r][c] = t[idx2];
                                        a[s][r][c + 1] = t[idx2 + 1];
                                        a[s][r][c + 2] = t[idx3];
                                        a[s][r][c + 3] = t[idx3 + 1];
                                        a[s][r][c + 4] = t[idx4];
                                        a[s][r][c + 5] = t[idx4 + 1];
                                        a[s][r][c + 6] = t[idx5];
                                        a[s][r][c + 7] = t[idx5 + 1];
                                    }
                                }
                            } else if (columns == 4) {
                                for (int r = 0; r < rows; r++) {
                                    idx2 = 2 * r;
                                    idx3 = 2 * rows + 2 * r;
                                    t[idx2] = a[s][r][0];
                                    t[idx2 + 1] = a[s][r][1];
                                    t[idx3] = a[s][r][2];
                                    t[idx3 + 1] = a[s][r][3];
                                }
                                fftRows.complexInverse(t, 0, scale);
                                fftRows.complexInverse(t, 2 * rows, scale);
                                for (int r = 0; r < rows; r++) {
                                    idx2 = 2 * r;
                                    idx3 = 2 * rows + 2 * r;
                                    a[s][r][0] = t[idx2];
                                    a[s][r][1] = t[idx2 + 1];
                                    a[s][r][2] = t[idx3];
                                    a[s][r][3] = t[idx3 + 1];
                                }
                            } else if (columns == 2) {
                                for (int r = 0; r < rows; r++) {
                                    idx2 = 2 * r;
                                    t[idx2] = a[s][r][0];
                                    t[idx2 + 1] = a[s][r][1];
                                }
                                fftRows.complexInverse(t, 0, scale);
                                for (int r = 0; r < rows; r++) {
                                    idx2 = 2 * r;
                                    a[s][r][0] = t[idx2];
                                    a[s][r][1] = t[idx2 + 1];
                                }
                            }
                            if (icr != 0) {
                                for (int r = 0; r < rows; r++) {
                                    fftColumns.realInverse(a[s][r], scale);
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
            Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void xdft3da_subth2(final int icr, final int isgn, final double[][][] a, final boolean scale)
    {
        int i;
        final int nthreads = min(ConcurrencyUtils.getNumberOfThreads(), slices);
        int nt = slices;
        if (nt < rows) {
            nt = rows;
        }
        nt *= 8;
        if (columns == 4) {
            nt >>= 1;
        } else if (columns < 4) {
            nt >>= 2;
        }
        final int ntf = nt;
        Future<?>[] futures = new Future[nthreads];
        for (i = 0; i < nthreads; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {
                public void run()
                {
                    int idx2, idx3, idx4, idx5;
                    double[] t = new double[ntf];
                    if (isgn == -1) {
                        for (int s = n0; s < slices; s += nthreads) {
                            if (icr == 0) {
                                for (int r = 0; r < rows; r++) {
                                    fftColumns.complexForward(a[s][r]);
                                }
                            } else {
                                for (int r = 0; r < rows; r++) {
                                    fftColumns.realForward(a[s][r]);
                                }
                            }
                            if (columns > 4) {
                                for (int c = 0; c < columns; c += 8) {
                                    for (int r = 0; r < rows; r++) {
                                        idx2 = 2 * r;
                                        idx3 = 2 * rows + 2 * r;
                                        idx4 = idx3 + 2 * rows;
                                        idx5 = idx4 + 2 * rows;
                                        t[idx2] = a[s][r][c];
                                        t[idx2 + 1] = a[s][r][c + 1];
                                        t[idx3] = a[s][r][c + 2];
                                        t[idx3 + 1] = a[s][r][c + 3];
                                        t[idx4] = a[s][r][c + 4];
                                        t[idx4 + 1] = a[s][r][c + 5];
                                        t[idx5] = a[s][r][c + 6];
                                        t[idx5 + 1] = a[s][r][c + 7];
                                    }
                                    fftRows.complexForward(t, 0);
                                    fftRows.complexForward(t, 2 * rows);
                                    fftRows.complexForward(t, 4 * rows);
                                    fftRows.complexForward(t, 6 * rows);
                                    for (int r = 0; r < rows; r++) {
                                        idx2 = 2 * r;
                                        idx3 = 2 * rows + 2 * r;
                                        idx4 = idx3 + 2 * rows;
                                        idx5 = idx4 + 2 * rows;
                                        a[s][r][c] = t[idx2];
                                        a[s][r][c + 1] = t[idx2 + 1];
                                        a[s][r][c + 2] = t[idx3];
                                        a[s][r][c + 3] = t[idx3 + 1];
                                        a[s][r][c + 4] = t[idx4];
                                        a[s][r][c + 5] = t[idx4 + 1];
                                        a[s][r][c + 6] = t[idx5];
                                        a[s][r][c + 7] = t[idx5 + 1];
                                    }
                                }
                            } else if (columns == 4) {
                                for (int r = 0; r < rows; r++) {
                                    idx2 = 2 * r;
                                    idx3 = 2 * rows + 2 * r;
                                    t[idx2] = a[s][r][0];
                                    t[idx2 + 1] = a[s][r][1];
                                    t[idx3] = a[s][r][2];
                                    t[idx3 + 1] = a[s][r][3];
                                }
                                fftRows.complexForward(t, 0);
                                fftRows.complexForward(t, 2 * rows);
                                for (int r = 0; r < rows; r++) {
                                    idx2 = 2 * r;
                                    idx3 = 2 * rows + 2 * r;
                                    a[s][r][0] = t[idx2];
                                    a[s][r][1] = t[idx2 + 1];
                                    a[s][r][2] = t[idx3];
                                    a[s][r][3] = t[idx3 + 1];
                                }
                            } else if (columns == 2) {
                                for (int r = 0; r < rows; r++) {
                                    idx2 = 2 * r;
                                    t[idx2] = a[s][r][0];
                                    t[idx2 + 1] = a[s][r][1];
                                }
                                fftRows.complexForward(t, 0);
                                for (int r = 0; r < rows; r++) {
                                    idx2 = 2 * r;
                                    a[s][r][0] = t[idx2];
                                    a[s][r][1] = t[idx2 + 1];
                                }
                            }

                        }
                    } else {
                        for (int s = n0; s < slices; s += nthreads) {
                            if (icr == 0) {
                                for (int r = 0; r < rows; r++) {
                                    fftColumns.complexInverse(a[s][r], scale);
                                }
                            } else {
                                for (int r = 0; r < rows; r++) {
                                    fftColumns.realInverse2(a[s][r], 0, scale);
                                }
                            }
                            if (columns > 4) {
                                for (int c = 0; c < columns; c += 8) {
                                    for (int r = 0; r < rows; r++) {
                                        idx2 = 2 * r;
                                        idx3 = 2 * rows + 2 * r;
                                        idx4 = idx3 + 2 * rows;
                                        idx5 = idx4 + 2 * rows;
                                        t[idx2] = a[s][r][c];
                                        t[idx2 + 1] = a[s][r][c + 1];
                                        t[idx3] = a[s][r][c + 2];
                                        t[idx3 + 1] = a[s][r][c + 3];
                                        t[idx4] = a[s][r][c + 4];
                                        t[idx4 + 1] = a[s][r][c + 5];
                                        t[idx5] = a[s][r][c + 6];
                                        t[idx5 + 1] = a[s][r][c + 7];
                                    }
                                    fftRows.complexInverse(t, 0, scale);
                                    fftRows.complexInverse(t, 2 * rows, scale);
                                    fftRows.complexInverse(t, 4 * rows, scale);
                                    fftRows.complexInverse(t, 6 * rows, scale);
                                    for (int r = 0; r < rows; r++) {
                                        idx2 = 2 * r;
                                        idx3 = 2 * rows + 2 * r;
                                        idx4 = idx3 + 2 * rows;
                                        idx5 = idx4 + 2 * rows;
                                        a[s][r][c] = t[idx2];
                                        a[s][r][c + 1] = t[idx2 + 1];
                                        a[s][r][c + 2] = t[idx3];
                                        a[s][r][c + 3] = t[idx3 + 1];
                                        a[s][r][c + 4] = t[idx4];
                                        a[s][r][c + 5] = t[idx4 + 1];
                                        a[s][r][c + 6] = t[idx5];
                                        a[s][r][c + 7] = t[idx5 + 1];
                                    }
                                }
                            } else if (columns == 4) {
                                for (int r = 0; r < rows; r++) {
                                    idx2 = 2 * r;
                                    idx3 = 2 * rows + 2 * r;
                                    t[idx2] = a[s][r][0];
                                    t[idx2 + 1] = a[s][r][1];
                                    t[idx3] = a[s][r][2];
                                    t[idx3 + 1] = a[s][r][3];
                                }
                                fftRows.complexInverse(t, 0, scale);
                                fftRows.complexInverse(t, 2 * rows, scale);
                                for (int r = 0; r < rows; r++) {
                                    idx2 = 2 * r;
                                    idx3 = 2 * rows + 2 * r;
                                    a[s][r][0] = t[idx2];
                                    a[s][r][1] = t[idx2 + 1];
                                    a[s][r][2] = t[idx3];
                                    a[s][r][3] = t[idx3 + 1];
                                }
                            } else if (columns == 2) {
                                for (int r = 0; r < rows; r++) {
                                    idx2 = 2 * r;
                                    t[idx2] = a[s][r][0];
                                    t[idx2 + 1] = a[s][r][1];
                                }
                                fftRows.complexInverse(t, 0, scale);
                                for (int r = 0; r < rows; r++) {
                                    idx2 = 2 * r;
                                    a[s][r][0] = t[idx2];
                                    a[s][r][1] = t[idx2 + 1];
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
            Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void cdft3db_subth(final int isgn, final double[] a, final boolean scale)
    {
        int i;
        final int nthreads = min(ConcurrencyUtils.getNumberOfThreads(), rows);
        int nt = slices;
        if (nt < rows) {
            nt = rows;
        }
        nt *= 8;
        if (columns == 4) {
            nt >>= 1;
        } else if (columns < 4) {
            nt >>= 2;
        }
        final int ntf = nt;
        Future<?>[] futures = new Future[nthreads];
        for (i = 0; i < nthreads; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {

                public void run()
                {
                    int idx0, idx1, idx2, idx3, idx4, idx5;
                    double[] t = new double[ntf];
                    if (isgn == -1) {
                        if (columns > 4) {
                            for (int r = n0; r < rows; r += nthreads) {
                                idx0 = r * rowStride;
                                for (int c = 0; c < columns; c += 8) {
                                    for (int s = 0; s < slices; s++) {
                                        idx1 = s * sliceStride + idx0 + c;
                                        idx2 = 2 * s;
                                        idx3 = 2 * slices + 2 * s;
                                        idx4 = idx3 + 2 * slices;
                                        idx5 = idx4 + 2 * slices;
                                        t[idx2] = a[idx1];
                                        t[idx2 + 1] = a[idx1 + 1];
                                        t[idx3] = a[idx1 + 2];
                                        t[idx3 + 1] = a[idx1 + 3];
                                        t[idx4] = a[idx1 + 4];
                                        t[idx4 + 1] = a[idx1 + 5];
                                        t[idx5] = a[idx1 + 6];
                                        t[idx5 + 1] = a[idx1 + 7];
                                    }
                                    fftSlices.complexForward(t, 0);
                                    fftSlices.complexForward(t, 2 * slices);
                                    fftSlices.complexForward(t, 4 * slices);
                                    fftSlices.complexForward(t, 6 * slices);
                                    for (int s = 0; s < slices; s++) {
                                        idx1 = s * sliceStride + idx0 + c;
                                        idx2 = 2 * s;
                                        idx3 = 2 * slices + 2 * s;
                                        idx4 = idx3 + 2 * slices;
                                        idx5 = idx4 + 2 * slices;
                                        a[idx1] = t[idx2];
                                        a[idx1 + 1] = t[idx2 + 1];
                                        a[idx1 + 2] = t[idx3];
                                        a[idx1 + 3] = t[idx3 + 1];
                                        a[idx1 + 4] = t[idx4];
                                        a[idx1 + 5] = t[idx4 + 1];
                                        a[idx1 + 6] = t[idx5];
                                        a[idx1 + 7] = t[idx5 + 1];
                                    }
                                }
                            }
                        } else if (columns == 4) {
                            for (int r = n0; r < rows; r += nthreads) {
                                idx0 = r * rowStride;
                                for (int s = 0; s < slices; s++) {
                                    idx1 = s * sliceStride + idx0;
                                    idx2 = 2 * s;
                                    idx3 = 2 * slices + 2 * s;
                                    t[idx2] = a[idx1];
                                    t[idx2 + 1] = a[idx1 + 1];
                                    t[idx3] = a[idx1 + 2];
                                    t[idx3 + 1] = a[idx1 + 3];
                                }
                                fftSlices.complexForward(t, 0);
                                fftSlices.complexForward(t, 2 * slices);
                                for (int s = 0; s < slices; s++) {
                                    idx1 = s * sliceStride + idx0;
                                    idx2 = 2 * s;
                                    idx3 = 2 * slices + 2 * s;
                                    a[idx1] = t[idx2];
                                    a[idx1 + 1] = t[idx2 + 1];
                                    a[idx1 + 2] = t[idx3];
                                    a[idx1 + 3] = t[idx3 + 1];
                                }
                            }
                        } else if (columns == 2) {
                            for (int r = n0; r < rows; r += nthreads) {
                                idx0 = r * rowStride;
                                for (int s = 0; s < slices; s++) {
                                    idx1 = s * sliceStride + idx0;
                                    idx2 = 2 * s;
                                    t[idx2] = a[idx1];
                                    t[idx2 + 1] = a[idx1 + 1];
                                }
                                fftSlices.complexForward(t, 0);
                                for (int s = 0; s < slices; s++) {
                                    idx1 = s * sliceStride + idx0;
                                    idx2 = 2 * s;
                                    a[idx1] = t[idx2];
                                    a[idx1 + 1] = t[idx2 + 1];
                                }
                            }
                        }
                    } else if (columns > 4) {
                        for (int r = n0; r < rows; r += nthreads) {
                            idx0 = r * rowStride;
                            for (int c = 0; c < columns; c += 8) {
                                for (int s = 0; s < slices; s++) {
                                    idx1 = s * sliceStride + idx0 + c;
                                    idx2 = 2 * s;
                                    idx3 = 2 * slices + 2 * s;
                                    idx4 = idx3 + 2 * slices;
                                    idx5 = idx4 + 2 * slices;
                                    t[idx2] = a[idx1];
                                    t[idx2 + 1] = a[idx1 + 1];
                                    t[idx3] = a[idx1 + 2];
                                    t[idx3 + 1] = a[idx1 + 3];
                                    t[idx4] = a[idx1 + 4];
                                    t[idx4 + 1] = a[idx1 + 5];
                                    t[idx5] = a[idx1 + 6];
                                    t[idx5 + 1] = a[idx1 + 7];
                                }
                                fftSlices.complexInverse(t, 0, scale);
                                fftSlices.complexInverse(t, 2 * slices, scale);
                                fftSlices.complexInverse(t, 4 * slices, scale);
                                fftSlices.complexInverse(t, 6 * slices, scale);
                                for (int s = 0; s < slices; s++) {
                                    idx1 = s * sliceStride + idx0 + c;
                                    idx2 = 2 * s;
                                    idx3 = 2 * slices + 2 * s;
                                    idx4 = idx3 + 2 * slices;
                                    idx5 = idx4 + 2 * slices;
                                    a[idx1] = t[idx2];
                                    a[idx1 + 1] = t[idx2 + 1];
                                    a[idx1 + 2] = t[idx3];
                                    a[idx1 + 3] = t[idx3 + 1];
                                    a[idx1 + 4] = t[idx4];
                                    a[idx1 + 5] = t[idx4 + 1];
                                    a[idx1 + 6] = t[idx5];
                                    a[idx1 + 7] = t[idx5 + 1];
                                }
                            }
                        }
                    } else if (columns == 4) {
                        for (int r = n0; r < rows; r += nthreads) {
                            idx0 = r * rowStride;
                            for (int s = 0; s < slices; s++) {
                                idx1 = s * sliceStride + idx0;
                                idx2 = 2 * s;
                                idx3 = 2 * slices + 2 * s;
                                t[idx2] = a[idx1];
                                t[idx2 + 1] = a[idx1 + 1];
                                t[idx3] = a[idx1 + 2];
                                t[idx3 + 1] = a[idx1 + 3];
                            }
                            fftSlices.complexInverse(t, 0, scale);
                            fftSlices.complexInverse(t, 2 * slices, scale);
                            for (int s = 0; s < slices; s++) {
                                idx1 = s * sliceStride + idx0;
                                idx2 = 2 * s;
                                idx3 = 2 * slices + 2 * s;
                                a[idx1] = t[idx2];
                                a[idx1 + 1] = t[idx2 + 1];
                                a[idx1 + 2] = t[idx3];
                                a[idx1 + 3] = t[idx3 + 1];
                            }
                        }
                    } else if (columns == 2) {
                        for (int r = n0; r < rows; r += nthreads) {
                            idx0 = r * rowStride;
                            for (int s = 0; s < slices; s++) {
                                idx1 = s * sliceStride + idx0;
                                idx2 = 2 * s;
                                t[idx2] = a[idx1];
                                t[idx2 + 1] = a[idx1 + 1];
                            }
                            fftSlices.complexInverse(t, 0, scale);
                            for (int s = 0; s < slices; s++) {
                                idx1 = s * sliceStride + idx0;
                                idx2 = 2 * s;
                                a[idx1] = t[idx2];
                                a[idx1 + 1] = t[idx2 + 1];
                            }
                        }
                    }

                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void cdft3db_subth(final int isgn, final DoubleLargeArray a, final boolean scale)
    {
        int i;
        final int nthreads = (int) min(ConcurrencyUtils.getNumberOfThreads(), rowsl);
        long nt = slicesl;
        if (nt < rowsl) {
            nt = rowsl;
        }
        nt *= 8;
        if (columnsl == 4) {
            nt >>= 1;
        } else if (columnsl < 4) {
            nt >>= 2;
        }
        final long ntf = nt;
        Future<?>[] futures = new Future[nthreads];
        for (i = 0; i < nthreads; i++) {
            final long n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {

                public void run()
                {
                    long idx0, idx1, idx2, idx3, idx4, idx5;
                    DoubleLargeArray t = new DoubleLargeArray(ntf);
                    if (isgn == -1) {
                        if (columnsl > 4) {
                            for (long r = n0; r < rowsl; r += nthreads) {
                                idx0 = r * rowStridel;
                                for (long c = 0; c < columnsl; c += 8) {
                                    for (long s = 0; s < slicesl; s++) {
                                        idx1 = s * sliceStridel + idx0 + c;
                                        idx2 = 2 * s;
                                        idx3 = 2 * slicesl + 2 * s;
                                        idx4 = idx3 + 2 * slicesl;
                                        idx5 = idx4 + 2 * slicesl;
                                        t.setDouble(idx2, a.getDouble(idx1));
                                        t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                                        t.setDouble(idx3, a.getDouble(idx1 + 2));
                                        t.setDouble(idx3 + 1, a.getDouble(idx1 + 3));
                                        t.setDouble(idx4, a.getDouble(idx1 + 4));
                                        t.setDouble(idx4 + 1, a.getDouble(idx1 + 5));
                                        t.setDouble(idx5, a.getDouble(idx1 + 6));
                                        t.setDouble(idx5 + 1, a.getDouble(idx1 + 7));
                                    }
                                    fftSlices.complexForward(t, 0);
                                    fftSlices.complexForward(t, 2 * slicesl);
                                    fftSlices.complexForward(t, 4 * slicesl);
                                    fftSlices.complexForward(t, 6 * slicesl);
                                    for (long s = 0; s < slicesl; s++) {
                                        idx1 = s * sliceStridel + idx0 + c;
                                        idx2 = 2 * s;
                                        idx3 = 2 * slicesl + 2 * s;
                                        idx4 = idx3 + 2 * slicesl;
                                        idx5 = idx4 + 2 * slicesl;
                                        a.setDouble(idx1, t.getDouble(idx2));
                                        a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                                        a.setDouble(idx1 + 2, t.getDouble(idx3));
                                        a.setDouble(idx1 + 3, t.getDouble(idx3 + 1));
                                        a.setDouble(idx1 + 4, t.getDouble(idx4));
                                        a.setDouble(idx1 + 5, t.getDouble(idx4 + 1));
                                        a.setDouble(idx1 + 6, t.getDouble(idx5));
                                        a.setDouble(idx1 + 7, t.getDouble(idx5 + 1));
                                    }
                                }
                            }
                        } else if (columnsl == 4) {
                            for (long r = n0; r < rowsl; r += nthreads) {
                                idx0 = r * rowStridel;
                                for (long s = 0; s < slicesl; s++) {
                                    idx1 = s * sliceStridel + idx0;
                                    idx2 = 2 * s;
                                    idx3 = 2 * slicesl + 2 * s;
                                    t.setDouble(idx2, a.getDouble(idx1));
                                    t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                                    t.setDouble(idx3, a.getDouble(idx1 + 2));
                                    t.setDouble(idx3 + 1, a.getDouble(idx1 + 3));
                                }
                                fftSlices.complexForward(t, 0);
                                fftSlices.complexForward(t, 2 * slicesl);
                                for (long s = 0; s < slicesl; s++) {
                                    idx1 = s * sliceStridel + idx0;
                                    idx2 = 2 * s;
                                    idx3 = 2 * slicesl + 2 * s;
                                    a.setDouble(idx1, t.getDouble(idx2));
                                    a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                                    a.setDouble(idx1 + 2, t.getDouble(idx3));
                                    a.setDouble(idx1 + 3, t.getDouble(idx3 + 1));
                                }
                            }
                        } else if (columnsl == 2) {
                            for (long r = n0; r < rowsl; r += nthreads) {
                                idx0 = r * rowStridel;
                                for (long s = 0; s < slicesl; s++) {
                                    idx1 = s * sliceStridel + idx0;
                                    idx2 = 2 * s;
                                    t.setDouble(idx2, a.getDouble(idx1));
                                    t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                                }
                                fftSlices.complexForward(t, 0);
                                for (long s = 0; s < slicesl; s++) {
                                    idx1 = s * sliceStridel + idx0;
                                    idx2 = 2 * s;
                                    a.setDouble(idx1, t.getDouble(idx2));
                                    a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                                }
                            }
                        }
                    } else if (columnsl > 4) {
                        for (long r = n0; r < rowsl; r += nthreads) {
                            idx0 = r * rowStridel;
                            for (long c = 0; c < columnsl; c += 8) {
                                for (long s = 0; s < slicesl; s++) {
                                    idx1 = s * sliceStridel + idx0 + c;
                                    idx2 = 2 * s;
                                    idx3 = 2 * slicesl + 2 * s;
                                    idx4 = idx3 + 2 * slicesl;
                                    idx5 = idx4 + 2 * slicesl;
                                    t.setDouble(idx2, a.getDouble(idx1));
                                    t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                                    t.setDouble(idx3, a.getDouble(idx1 + 2));
                                    t.setDouble(idx3 + 1, a.getDouble(idx1 + 3));
                                    t.setDouble(idx4, a.getDouble(idx1 + 4));
                                    t.setDouble(idx4 + 1, a.getDouble(idx1 + 5));
                                    t.setDouble(idx5, a.getDouble(idx1 + 6));
                                    t.setDouble(idx5 + 1, a.getDouble(idx1 + 7));
                                }
                                fftSlices.complexInverse(t, 0, scale);
                                fftSlices.complexInverse(t, 2 * slicesl, scale);
                                fftSlices.complexInverse(t, 4 * slicesl, scale);
                                fftSlices.complexInverse(t, 6 * slicesl, scale);
                                for (long s = 0; s < slicesl; s++) {
                                    idx1 = s * sliceStridel + idx0 + c;
                                    idx2 = 2 * s;
                                    idx3 = 2 * slicesl + 2 * s;
                                    idx4 = idx3 + 2 * slicesl;
                                    idx5 = idx4 + 2 * slicesl;
                                    a.setDouble(idx1, t.getDouble(idx2));
                                    a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                                    a.setDouble(idx1 + 2, t.getDouble(idx3));
                                    a.setDouble(idx1 + 3, t.getDouble(idx3 + 1));
                                    a.setDouble(idx1 + 4, t.getDouble(idx4));
                                    a.setDouble(idx1 + 5, t.getDouble(idx4 + 1));
                                    a.setDouble(idx1 + 6, t.getDouble(idx5));
                                    a.setDouble(idx1 + 7, t.getDouble(idx5 + 1));
                                }
                            }
                        }
                    } else if (columnsl == 4) {
                        for (long r = n0; r < rowsl; r += nthreads) {
                            idx0 = r * rowStridel;
                            for (long s = 0; s < slicesl; s++) {
                                idx1 = s * sliceStridel + idx0;
                                idx2 = 2 * s;
                                idx3 = 2 * slicesl + 2 * s;
                                t.setDouble(idx2, a.getDouble(idx1));
                                t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                                t.setDouble(idx3, a.getDouble(idx1 + 2));
                                t.setDouble(idx3 + 1, a.getDouble(idx1 + 3));
                            }
                            fftSlices.complexInverse(t, 0, scale);
                            fftSlices.complexInverse(t, 2 * slicesl, scale);
                            for (long s = 0; s < slicesl; s++) {
                                idx1 = s * sliceStridel + idx0;
                                idx2 = 2 * s;
                                idx3 = 2 * slicesl + 2 * s;
                                a.setDouble(idx1, t.getDouble(idx2));
                                a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                                a.setDouble(idx1 + 2, t.getDouble(idx3));
                                a.setDouble(idx1 + 3, t.getDouble(idx3 + 1));
                            }
                        }
                    } else if (columnsl == 2) {
                        for (long r = n0; r < rowsl; r += nthreads) {
                            idx0 = r * rowStridel;
                            for (long s = 0; s < slicesl; s++) {
                                idx1 = s * sliceStridel + idx0;
                                idx2 = 2 * s;
                                t.setDouble(idx2, a.getDouble(idx1));
                                t.setDouble(idx2 + 1, a.getDouble(idx1 + 1));
                            }
                            fftSlices.complexInverse(t, 0, scale);
                            for (long s = 0; s < slicesl; s++) {
                                idx1 = s * sliceStridel + idx0;
                                idx2 = 2 * s;
                                a.setDouble(idx1, t.getDouble(idx2));
                                a.setDouble(idx1 + 1, t.getDouble(idx2 + 1));
                            }
                        }
                    }

                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void cdft3db_subth(final int isgn, final double[][][] a, final boolean scale)
    {
        int i;
        final int nthreads = min(ConcurrencyUtils.getNumberOfThreads(), rows);
        int nt = slices;
        if (nt < rows) {
            nt = rows;
        }
        nt *= 8;
        if (columns == 4) {
            nt >>= 1;
        } else if (columns < 4) {
            nt >>= 2;
        }
        final int ntf = nt;
        Future<?>[] futures = new Future[nthreads];
        for (i = 0; i < nthreads; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {

                public void run()
                {
                    int idx2, idx3, idx4, idx5;
                    double[] t = new double[ntf];
                    if (isgn == -1) {
                        if (columns > 4) {
                            for (int r = n0; r < rows; r += nthreads) {
                                for (int c = 0; c < columns; c += 8) {
                                    for (int s = 0; s < slices; s++) {
                                        idx2 = 2 * s;
                                        idx3 = 2 * slices + 2 * s;
                                        idx4 = idx3 + 2 * slices;
                                        idx5 = idx4 + 2 * slices;
                                        t[idx2] = a[s][r][c];
                                        t[idx2 + 1] = a[s][r][c + 1];
                                        t[idx3] = a[s][r][c + 2];
                                        t[idx3 + 1] = a[s][r][c + 3];
                                        t[idx4] = a[s][r][c + 4];
                                        t[idx4 + 1] = a[s][r][c + 5];
                                        t[idx5] = a[s][r][c + 6];
                                        t[idx5 + 1] = a[s][r][c + 7];
                                    }
                                    fftSlices.complexForward(t, 0);
                                    fftSlices.complexForward(t, 2 * slices);
                                    fftSlices.complexForward(t, 4 * slices);
                                    fftSlices.complexForward(t, 6 * slices);
                                    for (int s = 0; s < slices; s++) {
                                        idx2 = 2 * s;
                                        idx3 = 2 * slices + 2 * s;
                                        idx4 = idx3 + 2 * slices;
                                        idx5 = idx4 + 2 * slices;
                                        a[s][r][c] = t[idx2];
                                        a[s][r][c + 1] = t[idx2 + 1];
                                        a[s][r][c + 2] = t[idx3];
                                        a[s][r][c + 3] = t[idx3 + 1];
                                        a[s][r][c + 4] = t[idx4];
                                        a[s][r][c + 5] = t[idx4 + 1];
                                        a[s][r][c + 6] = t[idx5];
                                        a[s][r][c + 7] = t[idx5 + 1];
                                    }
                                }
                            }
                        } else if (columns == 4) {
                            for (int r = n0; r < rows; r += nthreads) {
                                for (int s = 0; s < slices; s++) {
                                    idx2 = 2 * s;
                                    idx3 = 2 * slices + 2 * s;
                                    t[idx2] = a[s][r][0];
                                    t[idx2 + 1] = a[s][r][1];
                                    t[idx3] = a[s][r][2];
                                    t[idx3 + 1] = a[s][r][3];
                                }
                                fftSlices.complexForward(t, 0);
                                fftSlices.complexForward(t, 2 * slices);
                                for (int s = 0; s < slices; s++) {
                                    idx2 = 2 * s;
                                    idx3 = 2 * slices + 2 * s;
                                    a[s][r][0] = t[idx2];
                                    a[s][r][1] = t[idx2 + 1];
                                    a[s][r][2] = t[idx3];
                                    a[s][r][3] = t[idx3 + 1];
                                }
                            }
                        } else if (columns == 2) {
                            for (int r = n0; r < rows; r += nthreads) {
                                for (int s = 0; s < slices; s++) {
                                    idx2 = 2 * s;
                                    t[idx2] = a[s][r][0];
                                    t[idx2 + 1] = a[s][r][1];
                                }
                                fftSlices.complexForward(t, 0);
                                for (int s = 0; s < slices; s++) {
                                    idx2 = 2 * s;
                                    a[s][r][0] = t[idx2];
                                    a[s][r][1] = t[idx2 + 1];
                                }
                            }
                        }
                    } else if (columns > 4) {
                        for (int r = n0; r < rows; r += nthreads) {
                            for (int c = 0; c < columns; c += 8) {
                                for (int s = 0; s < slices; s++) {
                                    idx2 = 2 * s;
                                    idx3 = 2 * slices + 2 * s;
                                    idx4 = idx3 + 2 * slices;
                                    idx5 = idx4 + 2 * slices;
                                    t[idx2] = a[s][r][c];
                                    t[idx2 + 1] = a[s][r][c + 1];
                                    t[idx3] = a[s][r][c + 2];
                                    t[idx3 + 1] = a[s][r][c + 3];
                                    t[idx4] = a[s][r][c + 4];
                                    t[idx4 + 1] = a[s][r][c + 5];
                                    t[idx5] = a[s][r][c + 6];
                                    t[idx5 + 1] = a[s][r][c + 7];
                                }
                                fftSlices.complexInverse(t, 0, scale);
                                fftSlices.complexInverse(t, 2 * slices, scale);
                                fftSlices.complexInverse(t, 4 * slices, scale);
                                fftSlices.complexInverse(t, 6 * slices, scale);
                                for (int s = 0; s < slices; s++) {
                                    idx2 = 2 * s;
                                    idx3 = 2 * slices + 2 * s;
                                    idx4 = idx3 + 2 * slices;
                                    idx5 = idx4 + 2 * slices;
                                    a[s][r][c] = t[idx2];
                                    a[s][r][c + 1] = t[idx2 + 1];
                                    a[s][r][c + 2] = t[idx3];
                                    a[s][r][c + 3] = t[idx3 + 1];
                                    a[s][r][c + 4] = t[idx4];
                                    a[s][r][c + 5] = t[idx4 + 1];
                                    a[s][r][c + 6] = t[idx5];
                                    a[s][r][c + 7] = t[idx5 + 1];
                                }
                            }
                        }
                    } else if (columns == 4) {
                        for (int r = n0; r < rows; r += nthreads) {
                            for (int s = 0; s < slices; s++) {
                                idx2 = 2 * s;
                                idx3 = 2 * slices + 2 * s;
                                t[idx2] = a[s][r][0];
                                t[idx2 + 1] = a[s][r][1];
                                t[idx3] = a[s][r][2];
                                t[idx3 + 1] = a[s][r][3];
                            }
                            fftSlices.complexInverse(t, 0, scale);
                            fftSlices.complexInverse(t, 2 * slices, scale);
                            for (int s = 0; s < slices; s++) {
                                idx2 = 2 * s;
                                idx3 = 2 * slices + 2 * s;
                                a[s][r][0] = t[idx2];
                                a[s][r][1] = t[idx2 + 1];
                                a[s][r][2] = t[idx3];
                                a[s][r][3] = t[idx3 + 1];
                            }
                        }
                    } else if (columns == 2) {
                        for (int r = n0; r < rows; r += nthreads) {
                            for (int s = 0; s < slices; s++) {
                                idx2 = 2 * s;
                                t[idx2] = a[s][r][0];
                                t[idx2 + 1] = a[s][r][1];
                            }
                            fftSlices.complexInverse(t, 0, scale);
                            for (int s = 0; s < slices; s++) {
                                idx2 = 2 * s;
                                a[s][r][0] = t[idx2];
                                a[s][r][1] = t[idx2 + 1];
                            }
                        }
                    }

                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void rdft3d_sub(int isgn, double[] a)
    {
        int n1h, n2h, i, j, k, l, idx1, idx2, idx3, idx4;
        double xi;

        n1h = slices >> 1;
        n2h = rows >> 1;
        if (isgn < 0) {
            for (i = 1; i < n1h; i++) {
                j = slices - i;
                idx1 = i * sliceStride;
                idx2 = j * sliceStride;
                idx3 = i * sliceStride + n2h * rowStride;
                idx4 = j * sliceStride + n2h * rowStride;
                xi = a[idx1] - a[idx2];
                a[idx1] += a[idx2];
                a[idx2] = xi;
                xi = a[idx2 + 1] - a[idx1 + 1];
                a[idx1 + 1] += a[idx2 + 1];
                a[idx2 + 1] = xi;
                xi = a[idx3] - a[idx4];
                a[idx3] += a[idx4];
                a[idx4] = xi;
                xi = a[idx4 + 1] - a[idx3 + 1];
                a[idx3 + 1] += a[idx4 + 1];
                a[idx4 + 1] = xi;
                for (k = 1; k < n2h; k++) {
                    l = rows - k;
                    idx1 = i * sliceStride + k * rowStride;
                    idx2 = j * sliceStride + l * rowStride;
                    xi = a[idx1] - a[idx2];
                    a[idx1] += a[idx2];
                    a[idx2] = xi;
                    xi = a[idx2 + 1] - a[idx1 + 1];
                    a[idx1 + 1] += a[idx2 + 1];
                    a[idx2 + 1] = xi;
                    idx3 = j * sliceStride + k * rowStride;
                    idx4 = i * sliceStride + l * rowStride;
                    xi = a[idx3] - a[idx4];
                    a[idx3] += a[idx4];
                    a[idx4] = xi;
                    xi = a[idx4 + 1] - a[idx3 + 1];
                    a[idx3 + 1] += a[idx4 + 1];
                    a[idx4 + 1] = xi;
                }
            }
            for (k = 1; k < n2h; k++) {
                l = rows - k;
                idx1 = k * rowStride;
                idx2 = l * rowStride;
                xi = a[idx1] - a[idx2];
                a[idx1] += a[idx2];
                a[idx2] = xi;
                xi = a[idx2 + 1] - a[idx1 + 1];
                a[idx1 + 1] += a[idx2 + 1];
                a[idx2 + 1] = xi;
                idx3 = n1h * sliceStride + k * rowStride;
                idx4 = n1h * sliceStride + l * rowStride;
                xi = a[idx3] - a[idx4];
                a[idx3] += a[idx4];
                a[idx4] = xi;
                xi = a[idx4 + 1] - a[idx3 + 1];
                a[idx3 + 1] += a[idx4 + 1];
                a[idx4 + 1] = xi;
            }
        } else {
            for (i = 1; i < n1h; i++) {
                j = slices - i;
                idx1 = j * sliceStride;
                idx2 = i * sliceStride;
                a[idx1] = 0.5f * (a[idx2] - a[idx1]);
                a[idx2] -= a[idx1];
                a[idx1 + 1] = 0.5f * (a[idx2 + 1] + a[idx1 + 1]);
                a[idx2 + 1] -= a[idx1 + 1];
                idx3 = j * sliceStride + n2h * rowStride;
                idx4 = i * sliceStride + n2h * rowStride;
                a[idx3] = 0.5f * (a[idx4] - a[idx3]);
                a[idx4] -= a[idx3];
                a[idx3 + 1] = 0.5f * (a[idx4 + 1] + a[idx3 + 1]);
                a[idx4 + 1] -= a[idx3 + 1];
                for (k = 1; k < n2h; k++) {
                    l = rows - k;
                    idx1 = j * sliceStride + l * rowStride;
                    idx2 = i * sliceStride + k * rowStride;
                    a[idx1] = 0.5f * (a[idx2] - a[idx1]);
                    a[idx2] -= a[idx1];
                    a[idx1 + 1] = 0.5f * (a[idx2 + 1] + a[idx1 + 1]);
                    a[idx2 + 1] -= a[idx1 + 1];
                    idx3 = i * sliceStride + l * rowStride;
                    idx4 = j * sliceStride + k * rowStride;
                    a[idx3] = 0.5f * (a[idx4] - a[idx3]);
                    a[idx4] -= a[idx3];
                    a[idx3 + 1] = 0.5f * (a[idx4 + 1] + a[idx3 + 1]);
                    a[idx4 + 1] -= a[idx3 + 1];
                }
            }
            for (k = 1; k < n2h; k++) {
                l = rows - k;
                idx1 = l * rowStride;
                idx2 = k * rowStride;
                a[idx1] = 0.5f * (a[idx2] - a[idx1]);
                a[idx2] -= a[idx1];
                a[idx1 + 1] = 0.5f * (a[idx2 + 1] + a[idx1 + 1]);
                a[idx2 + 1] -= a[idx1 + 1];
                idx3 = n1h * sliceStride + l * rowStride;
                idx4 = n1h * sliceStride + k * rowStride;
                a[idx3] = 0.5f * (a[idx4] - a[idx3]);
                a[idx4] -= a[idx3];
                a[idx3 + 1] = 0.5f * (a[idx4 + 1] + a[idx3 + 1]);
                a[idx4 + 1] -= a[idx3 + 1];
            }
        }
    }

    private void rdft3d_sub(int isgn, DoubleLargeArray a)
    {
        long n1h, n2h, i, j, k, l, idx1, idx2, idx3, idx4;
        double xi;

        n1h = slicesl >> 1l;
        n2h = rowsl >> 1l;
        if (isgn < 0) {
            for (i = 1; i < n1h; i++) {
                j = slicesl - i;
                idx1 = i * sliceStridel;
                idx2 = j * sliceStridel;
                idx3 = i * sliceStridel + n2h * rowStridel;
                idx4 = j * sliceStridel + n2h * rowStridel;
                xi = a.getDouble(idx1) - a.getDouble(idx2);
                a.setDouble(idx1, a.getDouble(idx1) + a.getDouble(idx2));
                a.setDouble(idx2, xi);
                xi = a.getDouble(idx2 + 1) - a.getDouble(idx1 + 1);
                a.setDouble(idx1 + 1, a.getDouble(idx1 + 1) + a.getDouble(idx2 + 1));
                a.setDouble(idx2 + 1, xi);
                xi = a.getDouble(idx3) - a.getDouble(idx4);
                a.setDouble(idx3, a.getDouble(idx3) + a.getDouble(idx4));
                a.setDouble(idx4, xi);
                xi = a.getDouble(idx4 + 1) - a.getDouble(idx3 + 1);
                a.setDouble(idx3 + 1, a.getDouble(idx3 + 1) + a.getDouble(idx4 + 1));
                a.setDouble(idx4 + 1, xi);
                for (k = 1; k < n2h; k++) {
                    l = rowsl - k;
                    idx1 = i * sliceStridel + k * rowStridel;
                    idx2 = j * sliceStridel + l * rowStridel;
                    xi = a.getDouble(idx1) - a.getDouble(idx2);
                    a.setDouble(idx1, a.getDouble(idx1) + a.getDouble(idx2));
                    a.setDouble(idx2, xi);
                    xi = a.getDouble(idx2 + 1) - a.getDouble(idx1 + 1);
                    a.setDouble(idx1 + 1, a.getDouble(idx1 + 1) + a.getDouble(idx2 + 1));
                    a.setDouble(idx2 + 1, xi);
                    idx3 = j * sliceStridel + k * rowStridel;
                    idx4 = i * sliceStridel + l * rowStridel;
                    xi = a.getDouble(idx3) - a.getDouble(idx4);
                    a.setDouble(idx3, a.getDouble(idx3) + a.getDouble(idx4));
                    a.setDouble(idx4, xi);
                    xi = a.getDouble(idx4 + 1) - a.getDouble(idx3 + 1);
                    a.setDouble(idx3 + 1, a.getDouble(idx3 + 1) + a.getDouble(idx4 + 1));
                    a.setDouble(idx4 + 1, xi);
                }
            }
            for (k = 1; k < n2h; k++) {
                l = rowsl - k;
                idx1 = k * rowStridel;
                idx2 = l * rowStridel;
                xi = a.getDouble(idx1) - a.getDouble(idx2);
                a.setDouble(idx1, a.getDouble(idx1) + a.getDouble(idx2));
                a.setDouble(idx2, xi);
                xi = a.getDouble(idx2 + 1) - a.getDouble(idx1 + 1);
                a.setDouble(idx1 + 1, a.getDouble(idx1 + 1) + a.getDouble(idx2 + 1));
                a.setDouble(idx2 + 1, xi);
                idx3 = n1h * sliceStridel + k * rowStridel;
                idx4 = n1h * sliceStridel + l * rowStridel;
                xi = a.getDouble(idx3) - a.getDouble(idx4);
                a.setDouble(idx3, a.getDouble(idx3) + a.getDouble(idx4));
                a.setDouble(idx4, xi);
                xi = a.getDouble(idx4 + 1) - a.getDouble(idx3 + 1);
                a.setDouble(idx3 + 1, a.getDouble(idx3 + 1) + a.getDouble(idx4 + 1));
                a.setDouble(idx4 + 1, xi);
            }
        } else {
            for (i = 1; i < n1h; i++) {
                j = slicesl - i;
                idx1 = j * sliceStridel;
                idx2 = i * sliceStridel;
                a.setDouble(idx1, 0.5f * (a.getDouble(idx2) - a.getDouble(idx1)));
                a.setDouble(idx2, a.getDouble(idx2) - a.getDouble(idx1));
                a.setDouble(idx1 + 1, 0.5f * (a.getDouble(idx2 + 1) + a.getDouble(idx1 + 1)));
                a.setDouble(idx2 + 1, a.getDouble(idx2 + 1) - a.getDouble(idx1 + 1));
                idx3 = j * sliceStridel + n2h * rowStridel;
                idx4 = i * sliceStridel + n2h * rowStridel;
                a.setDouble(idx3, 0.5f * (a.getDouble(idx4) - a.getDouble(idx3)));
                a.setDouble(idx4, a.getDouble(idx4) - a.getDouble(idx3));
                a.setDouble(idx3 + 1, 0.5f * (a.getDouble(idx4 + 1) + a.getDouble(idx3 + 1)));
                a.setDouble(idx4 + 1, a.getDouble(idx4 + 1) - a.getDouble(idx3 + 1));
                for (k = 1; k < n2h; k++) {
                    l = rowsl - k;
                    idx1 = j * sliceStridel + l * rowStridel;
                    idx2 = i * sliceStridel + k * rowStridel;
                    a.setDouble(idx1, 0.5f * (a.getDouble(idx2) - a.getDouble(idx1)));
                    a.setDouble(idx2, a.getDouble(idx2) - a.getDouble(idx1));
                    a.setDouble(idx1 + 1, 0.5f * (a.getDouble(idx2 + 1) + a.getDouble(idx1 + 1)));
                    a.setDouble(idx2 + 1, a.getDouble(idx2 + 1) - a.getDouble(idx1 + 1));
                    idx3 = i * sliceStridel + l * rowStridel;
                    idx4 = j * sliceStridel + k * rowStridel;
                    a.setDouble(idx3, 0.5f * (a.getDouble(idx4) - a.getDouble(idx3)));
                    a.setDouble(idx4, a.getDouble(idx4) - a.getDouble(idx3));
                    a.setDouble(idx3 + 1, 0.5f * (a.getDouble(idx4 + 1) + a.getDouble(idx3 + 1)));
                    a.setDouble(idx4 + 1, a.getDouble(idx4 + 1) - a.getDouble(idx3 + 1));
                }
            }
            for (k = 1; k < n2h; k++) {
                l = rowsl - k;
                idx1 = l * rowStridel;
                idx2 = k * rowStridel;
                a.setDouble(idx1, 0.5f * (a.getDouble(idx2) - a.getDouble(idx1)));
                a.setDouble(idx2, a.getDouble(idx2) - a.getDouble(idx1));
                a.setDouble(idx1 + 1, 0.5f * (a.getDouble(idx2 + 1) + a.getDouble(idx1 + 1)));
                a.setDouble(idx2 + 1, a.getDouble(idx2 + 1) - a.getDouble(idx1 + 1));
                idx3 = n1h * sliceStridel + l * rowStridel;
                idx4 = n1h * sliceStridel + k * rowStridel;
                a.setDouble(idx3, 0.5f * (a.getDouble(idx4) - a.getDouble(idx3)));
                a.setDouble(idx4, a.getDouble(idx4) - a.getDouble(idx3));
                a.setDouble(idx3 + 1, 0.5f * (a.getDouble(idx4 + 1) + a.getDouble(idx3 + 1)));
                a.setDouble(idx4 + 1, a.getDouble(idx4 + 1) - a.getDouble(idx3 + 1));
            }
        }
    }

    private void rdft3d_sub(int isgn, double[][][] a)
    {
        int n1h, n2h, i, j, k, l;
        double xi;

        n1h = slices >> 1;
        n2h = rows >> 1;
        if (isgn < 0) {
            for (i = 1; i < n1h; i++) {
                j = slices - i;
                xi = a[i][0][0] - a[j][0][0];
                a[i][0][0] += a[j][0][0];
                a[j][0][0] = xi;
                xi = a[j][0][1] - a[i][0][1];
                a[i][0][1] += a[j][0][1];
                a[j][0][1] = xi;
                xi = a[i][n2h][0] - a[j][n2h][0];
                a[i][n2h][0] += a[j][n2h][0];
                a[j][n2h][0] = xi;
                xi = a[j][n2h][1] - a[i][n2h][1];
                a[i][n2h][1] += a[j][n2h][1];
                a[j][n2h][1] = xi;
                for (k = 1; k < n2h; k++) {
                    l = rows - k;
                    xi = a[i][k][0] - a[j][l][0];
                    a[i][k][0] += a[j][l][0];
                    a[j][l][0] = xi;
                    xi = a[j][l][1] - a[i][k][1];
                    a[i][k][1] += a[j][l][1];
                    a[j][l][1] = xi;
                    xi = a[j][k][0] - a[i][l][0];
                    a[j][k][0] += a[i][l][0];
                    a[i][l][0] = xi;
                    xi = a[i][l][1] - a[j][k][1];
                    a[j][k][1] += a[i][l][1];
                    a[i][l][1] = xi;
                }
            }
            for (k = 1; k < n2h; k++) {
                l = rows - k;
                xi = a[0][k][0] - a[0][l][0];
                a[0][k][0] += a[0][l][0];
                a[0][l][0] = xi;
                xi = a[0][l][1] - a[0][k][1];
                a[0][k][1] += a[0][l][1];
                a[0][l][1] = xi;
                xi = a[n1h][k][0] - a[n1h][l][0];
                a[n1h][k][0] += a[n1h][l][0];
                a[n1h][l][0] = xi;
                xi = a[n1h][l][1] - a[n1h][k][1];
                a[n1h][k][1] += a[n1h][l][1];
                a[n1h][l][1] = xi;
            }
        } else {
            for (i = 1; i < n1h; i++) {
                j = slices - i;
                a[j][0][0] = 0.5f * (a[i][0][0] - a[j][0][0]);
                a[i][0][0] -= a[j][0][0];
                a[j][0][1] = 0.5f * (a[i][0][1] + a[j][0][1]);
                a[i][0][1] -= a[j][0][1];
                a[j][n2h][0] = 0.5f * (a[i][n2h][0] - a[j][n2h][0]);
                a[i][n2h][0] -= a[j][n2h][0];
                a[j][n2h][1] = 0.5f * (a[i][n2h][1] + a[j][n2h][1]);
                a[i][n2h][1] -= a[j][n2h][1];
                for (k = 1; k < n2h; k++) {
                    l = rows - k;
                    a[j][l][0] = 0.5f * (a[i][k][0] - a[j][l][0]);
                    a[i][k][0] -= a[j][l][0];
                    a[j][l][1] = 0.5f * (a[i][k][1] + a[j][l][1]);
                    a[i][k][1] -= a[j][l][1];
                    a[i][l][0] = 0.5f * (a[j][k][0] - a[i][l][0]);
                    a[j][k][0] -= a[i][l][0];
                    a[i][l][1] = 0.5f * (a[j][k][1] + a[i][l][1]);
                    a[j][k][1] -= a[i][l][1];
                }
            }
            for (k = 1; k < n2h; k++) {
                l = rows - k;
                a[0][l][0] = 0.5f * (a[0][k][0] - a[0][l][0]);
                a[0][k][0] -= a[0][l][0];
                a[0][l][1] = 0.5f * (a[0][k][1] + a[0][l][1]);
                a[0][k][1] -= a[0][l][1];
                a[n1h][l][0] = 0.5f * (a[n1h][k][0] - a[n1h][l][0]);
                a[n1h][k][0] -= a[n1h][l][0];
                a[n1h][l][1] = 0.5f * (a[n1h][k][1] + a[n1h][l][1]);
                a[n1h][k][1] -= a[n1h][l][1];
            }
        }
    }

    private void fillSymmetric(final double[][][] a)
    {
        final int twon3 = 2 * columns;
        final int n2d2 = rows / 2;
        int n1d2 = slices / 2;
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && useThreads && (slices >= nthreads)) {
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
                            int idx1 = (slices - s) % slices;
                            for (int r = 0; r < rows; r++) {
                                int idx2 = (rows - r) % rows;
                                for (int c = 1; c < columns; c += 2) {
                                    int idx3 = twon3 - c;
                                    a[idx1][idx2][idx3] = -a[s][r][c + 2];
                                    a[idx1][idx2][idx3 - 1] = a[s][r][c + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            // ---------------------------------------------
            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (int s = firstSlice; s < lastSlice; s++) {
                            int idx1 = (slices - s) % slices;
                            for (int r = 1; r < n2d2; r++) {
                                int idx2 = rows - r;
                                a[idx1][r][columns] = a[s][idx2][1];
                                a[s][idx2][columns] = a[s][idx2][1];
                                a[idx1][r][columns + 1] = -a[s][idx2][0];
                                a[s][idx2][columns + 1] = a[s][idx2][0];
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (int s = firstSlice; s < lastSlice; s++) {
                            int idx1 = (slices - s) % slices;
                            for (int r = 1; r < n2d2; r++) {
                                int idx2 = rows - r;
                                a[idx1][idx2][0] = a[s][r][0];
                                a[idx1][idx2][1] = -a[s][r][1];
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

        } else {

            for (int s = 0; s < slices; s++) {
                int idx1 = (slices - s) % slices;
                for (int r = 0; r < rows; r++) {
                    int idx2 = (rows - r) % rows;
                    for (int c = 1; c < columns; c += 2) {
                        int idx3 = twon3 - c;
                        a[idx1][idx2][idx3] = -a[s][r][c + 2];
                        a[idx1][idx2][idx3 - 1] = a[s][r][c + 1];
                    }
                }
            }

            // ---------------------------------------------
            for (int s = 0; s < slices; s++) {
                int idx1 = (slices - s) % slices;
                for (int r = 1; r < n2d2; r++) {
                    int idx2 = rows - r;
                    a[idx1][r][columns] = a[s][idx2][1];
                    a[s][idx2][columns] = a[s][idx2][1];
                    a[idx1][r][columns + 1] = -a[s][idx2][0];
                    a[s][idx2][columns + 1] = a[s][idx2][0];
                }
            }

            for (int s = 0; s < slices; s++) {
                int idx1 = (slices - s) % slices;
                for (int r = 1; r < n2d2; r++) {
                    int idx2 = rows - r;
                    a[idx1][idx2][0] = a[s][r][0];
                    a[idx1][idx2][1] = -a[s][r][1];
                }
            }
        }

        // ----------------------------------------------------------
        for (int s = 1; s < n1d2; s++) {
            int idx1 = slices - s;
            a[s][0][columns] = a[idx1][0][1];
            a[idx1][0][columns] = a[idx1][0][1];
            a[s][0][columns + 1] = -a[idx1][0][0];
            a[idx1][0][columns + 1] = a[idx1][0][0];
            a[s][n2d2][columns] = a[idx1][n2d2][1];
            a[idx1][n2d2][columns] = a[idx1][n2d2][1];
            a[s][n2d2][columns + 1] = -a[idx1][n2d2][0];
            a[idx1][n2d2][columns + 1] = a[idx1][n2d2][0];
            a[idx1][0][0] = a[s][0][0];
            a[idx1][0][1] = -a[s][0][1];
            a[idx1][n2d2][0] = a[s][n2d2][0];
            a[idx1][n2d2][1] = -a[s][n2d2][1];

        }
        // ----------------------------------------

        a[0][0][columns] = a[0][0][1];
        a[0][0][1] = 0;
        a[0][n2d2][columns] = a[0][n2d2][1];
        a[0][n2d2][1] = 0;
        a[n1d2][0][columns] = a[n1d2][0][1];
        a[n1d2][0][1] = 0;
        a[n1d2][n2d2][columns] = a[n1d2][n2d2][1];
        a[n1d2][n2d2][1] = 0;
        a[n1d2][0][columns + 1] = 0;
        a[n1d2][n2d2][columns + 1] = 0;
    }

    private void fillSymmetric(final double[] a)
    {
        final int twon3 = 2 * columns;
        final int n2d2 = rows / 2;
        int n1d2 = slices / 2;

        final int twoSliceStride = rows * twon3;
        final int twoRowStride = twon3;

        int idx1, idx2, idx3, idx4, idx5, idx6;

        for (int s = (slices - 1); s >= 1; s--) {
            idx3 = s * sliceStride;
            idx4 = 2 * idx3;
            for (int r = 0; r < rows; r++) {
                idx5 = r * rowStride;
                idx6 = 2 * idx5;
                for (int c = 0; c < columns; c += 2) {
                    idx1 = idx3 + idx5 + c;
                    idx2 = idx4 + idx6 + c;
                    a[idx2] = a[idx1];
                    a[idx1] = 0;
                    idx1++;
                    idx2++;
                    a[idx2] = a[idx1];
                    a[idx1] = 0;
                }
            }
        }

        for (int r = 1; r < rows; r++) {
            idx3 = (rows - r) * rowStride;
            idx4 = (rows - r) * twoRowStride;
            for (int c = 0; c < columns; c += 2) {
                idx1 = idx3 + c;
                idx2 = idx4 + c;
                a[idx2] = a[idx1];
                a[idx1] = 0;
                idx1++;
                idx2++;
                a[idx2] = a[idx1];
                a[idx1] = 0;
            }
        }

        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && useThreads && (slices >= nthreads)) {
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
                            int idx3 = ((slices - s) % slices) * twoSliceStride;
                            int idx5 = s * twoSliceStride;
                            for (int r = 0; r < rows; r++) {
                                int idx4 = ((rows - r) % rows) * twoRowStride;
                                int idx6 = r * twoRowStride;
                                for (int c = 1; c < columns; c += 2) {
                                    int idx1 = idx3 + idx4 + twon3 - c;
                                    int idx2 = idx5 + idx6 + c;
                                    a[idx1] = -a[idx2 + 2];
                                    a[idx1 - 1] = a[idx2 + 1];
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            // ---------------------------------------------
            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (int s = firstSlice; s < lastSlice; s++) {
                            int idx5 = ((slices - s) % slices) * twoSliceStride;
                            int idx6 = s * twoSliceStride;
                            for (int r = 1; r < n2d2; r++) {
                                int idx4 = idx6 + (rows - r) * twoRowStride;
                                int idx1 = idx5 + r * twoRowStride + columns;
                                int idx2 = idx4 + columns;
                                int idx3 = idx4 + 1;
                                a[idx1] = a[idx3];
                                a[idx2] = a[idx3];
                                a[idx1 + 1] = -a[idx4];
                                a[idx2 + 1] = a[idx4];

                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }
            for (int l = 0; l < nthreads; l++) {
                final int firstSlice = l * p;
                final int lastSlice = (l == (nthreads - 1)) ? slices : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (int s = firstSlice; s < lastSlice; s++) {
                            int idx3 = ((slices - s) % slices) * twoSliceStride;
                            int idx4 = s * twoSliceStride;
                            for (int r = 1; r < n2d2; r++) {
                                int idx1 = idx3 + (rows - r) * twoRowStride;
                                int idx2 = idx4 + r * twoRowStride;
                                a[idx1] = a[idx2];
                                a[idx1 + 1] = -a[idx2 + 1];

                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {

            // -----------------------------------------------
            for (int s = 0; s < slices; s++) {
                idx3 = ((slices - s) % slices) * twoSliceStride;
                idx5 = s * twoSliceStride;
                for (int r = 0; r < rows; r++) {
                    idx4 = ((rows - r) % rows) * twoRowStride;
                    idx6 = r * twoRowStride;
                    for (int c = 1; c < columns; c += 2) {
                        idx1 = idx3 + idx4 + twon3 - c;
                        idx2 = idx5 + idx6 + c;
                        a[idx1] = -a[idx2 + 2];
                        a[idx1 - 1] = a[idx2 + 1];
                    }
                }
            }

            // ---------------------------------------------
            for (int s = 0; s < slices; s++) {
                idx5 = ((slices - s) % slices) * twoSliceStride;
                idx6 = s * twoSliceStride;
                for (int r = 1; r < n2d2; r++) {
                    idx4 = idx6 + (rows - r) * twoRowStride;
                    idx1 = idx5 + r * twoRowStride + columns;
                    idx2 = idx4 + columns;
                    idx3 = idx4 + 1;
                    a[idx1] = a[idx3];
                    a[idx2] = a[idx3];
                    a[idx1 + 1] = -a[idx4];
                    a[idx2 + 1] = a[idx4];

                }
            }

            for (int s = 0; s < slices; s++) {
                idx3 = ((slices - s) % slices) * twoSliceStride;
                idx4 = s * twoSliceStride;
                for (int r = 1; r < n2d2; r++) {
                    idx1 = idx3 + (rows - r) * twoRowStride;
                    idx2 = idx4 + r * twoRowStride;
                    a[idx1] = a[idx2];
                    a[idx1 + 1] = -a[idx2 + 1];

                }
            }
        }

        // ----------------------------------------------------------
        for (int s = 1; s < n1d2; s++) {
            idx1 = s * twoSliceStride;
            idx2 = (slices - s) * twoSliceStride;
            idx3 = n2d2 * twoRowStride;
            idx4 = idx1 + idx3;
            idx5 = idx2 + idx3;
            a[idx1 + columns] = a[idx2 + 1];
            a[idx2 + columns] = a[idx2 + 1];
            a[idx1 + columns + 1] = -a[idx2];
            a[idx2 + columns + 1] = a[idx2];
            a[idx4 + columns] = a[idx5 + 1];
            a[idx5 + columns] = a[idx5 + 1];
            a[idx4 + columns + 1] = -a[idx5];
            a[idx5 + columns + 1] = a[idx5];
            a[idx2] = a[idx1];
            a[idx2 + 1] = -a[idx1 + 1];
            a[idx5] = a[idx4];
            a[idx5 + 1] = -a[idx4 + 1];

        }

        // ----------------------------------------
        a[columns] = a[1];
        a[1] = 0;
        idx1 = n2d2 * twoRowStride;
        idx2 = n1d2 * twoSliceStride;
        idx3 = idx1 + idx2;
        a[idx1 + columns] = a[idx1 + 1];
        a[idx1 + 1] = 0;
        a[idx2 + columns] = a[idx2 + 1];
        a[idx2 + 1] = 0;
        a[idx3 + columns] = a[idx3 + 1];
        a[idx3 + 1] = 0;
        a[idx2 + columns + 1] = 0;
        a[idx3 + columns + 1] = 0;
    }

    private void fillSymmetric(final DoubleLargeArray a)
    {
        final long twon3 = 2 * columnsl;
        final long n2d2 = rowsl / 2;
        long n1d2 = slicesl / 2;

        final long twoSliceStride = rowsl * twon3;
        final long twoRowStride = twon3;

        long idx1, idx2, idx3, idx4, idx5, idx6;

        for (long s = (slicesl - 1); s >= 1; s--) {
            idx3 = s * sliceStridel;
            idx4 = 2 * idx3;
            for (long r = 0; r < rowsl; r++) {
                idx5 = r * rowStridel;
                idx6 = 2 * idx5;
                for (long c = 0; c < columnsl; c += 2) {
                    idx1 = idx3 + idx5 + c;
                    idx2 = idx4 + idx6 + c;
                    a.setDouble(idx2, a.getDouble(idx1));
                    a.setDouble(idx1, 0);
                    idx1++;
                    idx2++;
                    a.setDouble(idx2, a.getDouble(idx1));
                    a.setDouble(idx1, 0);
                }
            }
        }

        for (long r = 1; r < rowsl; r++) {
            idx3 = (rowsl - r) * rowStridel;
            idx4 = (rowsl - r) * twoRowStride;
            for (long c = 0; c < columnsl; c += 2) {
                idx1 = idx3 + c;
                idx2 = idx4 + c;
                a.setDouble(idx2, a.getDouble(idx1));
                a.setDouble(idx1, 0);
                idx1++;
                idx2++;
                a.setDouble(idx2, a.getDouble(idx1));
                a.setDouble(idx1, 0);
            }
        }

        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && useThreads && (slicesl >= nthreads)) {
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
                            long idx3 = ((slicesl - s) % slicesl) * twoSliceStride;
                            long idx5 = s * twoSliceStride;
                            for (long r = 0; r < rowsl; r++) {
                                long idx4 = ((rowsl - r) % rowsl) * twoRowStride;
                                long idx6 = r * twoRowStride;
                                for (long c = 1; c < columnsl; c += 2) {
                                    long idx1 = idx3 + idx4 + twon3 - c;
                                    long idx2 = idx5 + idx6 + c;
                                    a.setDouble(idx1, -a.getDouble(idx2 + 2));
                                    a.setDouble(idx1 - 1, a.getDouble(idx2 + 1));
                                }
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }

            // ---------------------------------------------
            for (int l = 0; l < nthreads; l++) {
                final long firstSlice = l * p;
                final long lastSlice = (l == (nthreads - 1)) ? slicesl : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (long s = firstSlice; s < lastSlice; s++) {
                            long idx5 = ((slicesl - s) % slicesl) * twoSliceStride;
                            long idx6 = s * twoSliceStride;
                            for (long r = 1; r < n2d2; r++) {
                                long idx4 = idx6 + (rowsl - r) * twoRowStride;
                                long idx1 = idx5 + r * twoRowStride + columnsl;
                                long idx2 = idx4 + columnsl;
                                long idx3 = idx4 + 1;
                                a.setDouble(idx1, a.getDouble(idx3));
                                a.setDouble(idx2, a.getDouble(idx3));
                                a.setDouble(idx1 + 1, -a.getDouble(idx4));
                                a.setDouble(idx2 + 1, a.getDouble(idx4));

                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }
            for (int l = 0; l < nthreads; l++) {
                final long firstSlice = l * p;
                final long lastSlice = (l == (nthreads - 1)) ? slicesl : firstSlice + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (long s = firstSlice; s < lastSlice; s++) {
                            long idx3 = ((slicesl - s) % slicesl) * twoSliceStride;
                            long idx4 = s * twoSliceStride;
                            for (long r = 1; r < n2d2; r++) {
                                long idx1 = idx3 + (rowsl - r) * twoRowStride;
                                long idx2 = idx4 + r * twoRowStride;
                                a.setDouble(idx1, a.getDouble(idx2));
                                a.setDouble(idx1 + 1, -a.getDouble(idx2 + 1));

                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(DoubleFFT_3D.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {

            // -----------------------------------------------
            for (long s = 0; s < slicesl; s++) {
                idx3 = ((slicesl - s) % slicesl) * twoSliceStride;
                idx5 = s * twoSliceStride;
                for (long r = 0; r < rowsl; r++) {
                    idx4 = ((rowsl - r) % rowsl) * twoRowStride;
                    idx6 = r * twoRowStride;
                    for (long c = 1; c < columnsl; c += 2) {
                        idx1 = idx3 + idx4 + twon3 - c;
                        idx2 = idx5 + idx6 + c;
                        a.setDouble(idx1, -a.getDouble(idx2 + 2));
                        a.setDouble(idx1 - 1, a.getDouble(idx2 + 1));
                    }
                }
            }

            // ---------------------------------------------
            for (long s = 0; s < slicesl; s++) {
                idx5 = ((slicesl - s) % slicesl) * twoSliceStride;
                idx6 = s * twoSliceStride;
                for (long r = 1; r < n2d2; r++) {
                    idx4 = idx6 + (rowsl - r) * twoRowStride;
                    idx1 = idx5 + r * twoRowStride + columnsl;
                    idx2 = idx4 + columnsl;
                    idx3 = idx4 + 1;
                    a.setDouble(idx1, a.getDouble(idx3));
                    a.setDouble(idx2, a.getDouble(idx3));
                    a.setDouble(idx1 + 1, -a.getDouble(idx4));
                    a.setDouble(idx2 + 1, a.getDouble(idx4));

                }
            }

            for (long s = 0; s < slicesl; s++) {
                idx3 = ((slicesl - s) % slicesl) * twoSliceStride;
                idx4 = s * twoSliceStride;
                for (long r = 1; r < n2d2; r++) {
                    idx1 = idx3 + (rowsl - r) * twoRowStride;
                    idx2 = idx4 + r * twoRowStride;
                    a.setDouble(idx1, a.getDouble(idx2));
                    a.setDouble(idx1 + 1, -a.getDouble(idx2 + 1));

                }
            }
        }

        // ----------------------------------------------------------
        for (long s = 1; s < n1d2; s++) {
            idx1 = s * twoSliceStride;
            idx2 = (slicesl - s) * twoSliceStride;
            idx3 = n2d2 * twoRowStride;
            idx4 = idx1 + idx3;
            idx5 = idx2 + idx3;
            a.setDouble(idx1 + columnsl, a.getDouble(idx2 + 1));
            a.setDouble(idx2 + columnsl, a.getDouble(idx2 + 1));
            a.setDouble(idx1 + columnsl + 1, -a.getDouble(idx2));
            a.setDouble(idx2 + columnsl + 1, a.getDouble(idx2));
            a.setDouble(idx4 + columnsl, a.getDouble(idx5 + 1));
            a.setDouble(idx5 + columnsl, a.getDouble(idx5 + 1));
            a.setDouble(idx4 + columnsl + 1, -a.getDouble(idx5));
            a.setDouble(idx5 + columnsl + 1, a.getDouble(idx5));
            a.setDouble(idx2, a.getDouble(idx1));
            a.setDouble(idx2 + 1, -a.getDouble(idx1 + 1));
            a.setDouble(idx5, a.getDouble(idx4));
            a.setDouble(idx5 + 1, -a.getDouble(idx4 + 1));

        }

        // ----------------------------------------
        a.setDouble(columnsl, a.getDouble(1));
        a.setDouble(1, 0);
        idx1 = n2d2 * twoRowStride;
        idx2 = n1d2 * twoSliceStride;
        idx3 = idx1 + idx2;
        a.setDouble(idx1 + columnsl, a.getDouble(idx1 + 1));
        a.setDouble(idx1 + 1, 0);
        a.setDouble(idx2 + columnsl, a.getDouble(idx2 + 1));
        a.setDouble(idx2 + 1, 0);
        a.setDouble(idx3 + columnsl, a.getDouble(idx3 + 1));
        a.setDouble(idx3 + 1, 0);
        a.setDouble(idx2 + columnsl + 1, 0);
        a.setDouble(idx3 + columnsl + 1, 0);
    }
}
