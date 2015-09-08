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
import pl.edu.icm.jlargearrays.FloatLargeArray;
import pl.edu.icm.jlargearrays.LargeArray;
import static org.apache.commons.math3.util.FastMath.*;

/**
 * Computes 2D Discrete Fourier Transform (DFT) of complex and real, single
 * precision data. The sizes of both dimensions can be arbitrary numbers. This
 * is a parallel implementation of split-radix and mixed-radix algorithms
 * optimized for SMP systems. <br>
 * <br>
 * Part of the code is derived from General Purpose FFT Package written by
 * Takuya Ooura (http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html)
 *  
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class FloatFFT_2D
{

    private int rows;

    private int columns;

    private long rowsl;

    private long columnsl;

    private FloatFFT_1D fftColumns, fftRows;

    private boolean isPowerOfTwo = false;

    private boolean useThreads = false;

    /**
     * Creates new instance of FloatFFT_2D.
     *  
     * @param rows    number of rows
     * @param columns number of columns
     */
    public FloatFFT_2D(long rows, long columns)
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
        CommonUtils.setUseLargeArrays(2 * rows * columns > LargeArray.getMaxSizeOf32bitArray());
        fftRows = new FloatFFT_1D(rows);
        if (rows == columns) {
            fftColumns = fftRows;
        } else {
            fftColumns = new FloatFFT_1D(columns);
        }
    }

    /**
     * Computes 2D forward DFT of complex data leaving the result in
     * <code>a</code>. The data is stored in 1D array in row-major order.
     * Complex number is stored as two float values in sequence: the real and
     * imaginary part, i.e. the input array must be of size rows*2*columns. The
     * physical layout of the input data has to be as follows:<br>
     *  
     * <pre>
     * a[k1*2*columns+2*k2] = Re[k1][k2],
     * a[k1*2*columns+2*k2+1] = Im[k1][k2], 0&lt;=k1&lt;rows, 0&lt;=k2&lt;columns,
     * </pre>
     *  
     * @param a data to transform
     */
    public void complexForward(final float[] a)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            columns = 2 * columns;
            if ((nthreads > 1) && useThreads) {
                xdft2d0_subth1(0, -1, a, true);
                cdft2d_subth(-1, a, true);
            } else {
                for (int r = 0; r < rows; r++) {
                    fftColumns.complexForward(a, r * columns);
                }
                cdft2d_sub(-1, a, true);
            }
            columns = columns / 2;
        } else {
            final int rowStride = 2 * columns;
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
                            for (int r = firstRow; r < lastRow; r++) {
                                fftColumns.complexForward(a, r * rowStride);
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
                }
                p = columns / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final int firstColumn = l * p;
                    final int lastColumn = (l == (nthreads - 1)) ? columns : firstColumn + p;
                    futures[l] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            float[] temp = new float[2 * rows];
                            for (int c = firstColumn; c < lastColumn; c++) {
                                int idx0 = 2 * c;
                                for (int r = 0; r < rows; r++) {
                                    int idx1 = 2 * r;
                                    int idx2 = r * rowStride + idx0;
                                    temp[idx1] = a[idx2];
                                    temp[idx1 + 1] = a[idx2 + 1];
                                }
                                fftRows.complexForward(temp);
                                for (int r = 0; r < rows; r++) {
                                    int idx1 = 2 * r;
                                    int idx2 = r * rowStride + idx0;
                                    a[idx2] = temp[idx1];
                                    a[idx2 + 1] = temp[idx1 + 1];
                                }
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else {
                for (int r = 0; r < rows; r++) {
                    fftColumns.complexForward(a, r * rowStride);
                }
                float[] temp = new float[2 * rows];
                for (int c = 0; c < columns; c++) {
                    int idx0 = 2 * c;
                    for (int r = 0; r < rows; r++) {
                        int idx1 = 2 * r;
                        int idx2 = r * rowStride + idx0;
                        temp[idx1] = a[idx2];
                        temp[idx1 + 1] = a[idx2 + 1];
                    }
                    fftRows.complexForward(temp);
                    for (int r = 0; r < rows; r++) {
                        int idx1 = 2 * r;
                        int idx2 = r * rowStride + idx0;
                        a[idx2] = temp[idx1];
                        a[idx2 + 1] = temp[idx1 + 1];
                    }
                }
            }
        }
    }

    /**
     * Computes 2D forward DFT of complex data leaving the result in
     * <code>a</code>. The data is stored in 1D array in row-major order.
     * Complex number is stored as two float values in sequence: the real and
     * imaginary part, i.e. the input array must be of size rows*2*columns. The
     * physical layout of the input data has to be as follows:<br>
     *  
     * <pre>
     * a[k1*2*columns+2*k2] = Re[k1][k2],
     * a[k1*2*columns+2*k2+1] = Im[k1][k2], 0&lt;=k1&lt;rows, 0&lt;=k2&lt;columns,
     * </pre>
     *  
     * @param a data to transform
     */
    public void complexForward(final FloatLargeArray a)
    {
        if (!a.isLarge() && !a.isConstant()) {
            complexForward(a.getData());
        } else {
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if (isPowerOfTwo) {
                columnsl = 2 * columnsl;
                if ((nthreads > 1) && useThreads) {
                    xdft2d0_subth1(0, -1, a, true);
                    cdft2d_subth(-1, a, true);
                } else {
                    for (int r = 0; r < rowsl; r++) {
                        fftColumns.complexForward(a, r * columnsl);
                    }
                    cdft2d_sub(-1, a, true);
                }
                columnsl = columnsl / 2;
            } else {
                final long rowStride = 2 * columnsl;
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
                                for (long r = firstRow; r < lastRow; r++) {
                                    fftColumns.complexForward(a, r * rowStride);
                                }
                            }
                        });
                    }
                    try {
                        ConcurrencyUtils.waitForCompletion(futures);
                    } catch (InterruptedException ex) {
                        Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
                    } catch (ExecutionException ex) {
                        Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
                    }
                    p = columnsl / nthreads;
                    for (int l = 0; l < nthreads; l++) {
                        final long firstColumn = l * p;
                        final long lastColumn = (l == (nthreads - 1)) ? columnsl : firstColumn + p;
                        futures[l] = ConcurrencyUtils.submit(new Runnable()
                        {
                            public void run()
                            {
                                FloatLargeArray temp = new FloatLargeArray(2 * rowsl, false);
                                for (long c = firstColumn; c < lastColumn; c++) {
                                    long idx0 = 2 * c;
                                    for (long r = 0; r < rowsl; r++) {
                                        long idx1 = 2 * r;
                                        long idx2 = r * rowStride + idx0;
                                        temp.setDouble(idx1, a.getFloat(idx2));
                                        temp.setDouble(idx1 + 1, a.getFloat(idx2 + 1));
                                    }
                                    fftRows.complexForward(temp);
                                    for (long r = 0; r < rowsl; r++) {
                                        long idx1 = 2 * r;
                                        long idx2 = r * rowStride + idx0;
                                        a.setDouble(idx2, temp.getFloat(idx1));
                                        a.setDouble(idx2 + 1, temp.getFloat(idx1 + 1));
                                    }
                                }
                            }
                        });
                    }
                    try {
                        ConcurrencyUtils.waitForCompletion(futures);
                    } catch (InterruptedException ex) {
                        Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
                    } catch (ExecutionException ex) {
                        Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
                    }
                } else {
                    for (long r = 0; r < rowsl; r++) {
                        fftColumns.complexForward(a, r * rowStride);
                    }
                    FloatLargeArray temp = new FloatLargeArray(2 * rowsl, false);
                    for (long c = 0; c < columnsl; c++) {
                        long idx0 = 2 * c;
                        for (long r = 0; r < rowsl; r++) {
                            long idx1 = 2 * r;
                            long idx2 = r * rowStride + idx0;
                            temp.setDouble(idx1, a.getFloat(idx2));
                            temp.setDouble(idx1 + 1, a.getFloat(idx2 + 1));
                        }
                        fftRows.complexForward(temp);
                        for (long r = 0; r < rowsl; r++) {
                            long idx1 = 2 * r;
                            long idx2 = r * rowStride + idx0;
                            a.setDouble(idx2, temp.getFloat(idx1));
                            a.setDouble(idx2 + 1, temp.getFloat(idx1 + 1));
                        }
                    }
                }
            }
        }
    }

    /**
     * Computes 2D forward DFT of complex data leaving the result in
     * <code>a</code>. The data is stored in 2D array. Complex data is
     * represented by 2 float values in sequence: the real and imaginary part,
     * i.e. the input array must be of size rows by 2*columns. The physical
     * layout of the input data has to be as follows:<br>
     *  
     * <pre>
     * a[k1][2*k2] = Re[k1][k2],
     * a[k1][2*k2+1] = Im[k1][k2], 0&lt;=k1&lt;rows, 0&lt;=k2&lt;columns,
     * </pre>
     *  
     * @param a data to transform
     */
    public void complexForward(final float[][] a)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            columns = 2 * columns;
            if ((nthreads > 1) && useThreads) {
                xdft2d0_subth1(0, -1, a, true);
                cdft2d_subth(-1, a, true);
            } else {
                for (int r = 0; r < rows; r++) {
                    fftColumns.complexForward(a[r]);
                }
                cdft2d_sub(-1, a, true);
            }
            columns = columns / 2;
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
                            fftColumns.complexForward(a[r]);
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }
            p = columns / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstColumn = l * p;
                final int lastColumn = (l == (nthreads - 1)) ? columns : firstColumn + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        float[] temp = new float[2 * rows];
                        for (int c = firstColumn; c < lastColumn; c++) {
                            int idx1 = 2 * c;
                            for (int r = 0; r < rows; r++) {
                                int idx2 = 2 * r;
                                temp[idx2] = a[r][idx1];
                                temp[idx2 + 1] = a[r][idx1 + 1];
                            }
                            fftRows.complexForward(temp);
                            for (int r = 0; r < rows; r++) {
                                int idx2 = 2 * r;
                                a[r][idx1] = temp[idx2];
                                a[r][idx1 + 1] = temp[idx2 + 1];
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {
            for (int r = 0; r < rows; r++) {
                fftColumns.complexForward(a[r]);
            }
            float[] temp = new float[2 * rows];
            for (int c = 0; c < columns; c++) {
                int idx1 = 2 * c;
                for (int r = 0; r < rows; r++) {
                    int idx2 = 2 * r;
                    temp[idx2] = a[r][idx1];
                    temp[idx2 + 1] = a[r][idx1 + 1];
                }
                fftRows.complexForward(temp);
                for (int r = 0; r < rows; r++) {
                    int idx2 = 2 * r;
                    a[r][idx1] = temp[idx2];
                    a[r][idx1 + 1] = temp[idx2 + 1];
                }
            }
        }
    }

    /**
     * Computes 2D inverse DFT of complex data leaving the result in
     * <code>a</code>. The data is stored in 1D array in row-major order.
     * Complex number is stored as two float values in sequence: the real and
     * imaginary part, i.e. the input array must be of size rows*2*columns. The
     * physical layout of the input data has to be as follows:<br>
     *  
     * <pre>
     * a[k1*2*columns+2*k2] = Re[k1][k2],
     * a[k1*2*columns+2*k2+1] = Im[k1][k2], 0&lt;=k1&lt;rows, 0&lt;=k2&lt;columns,
     * </pre>
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     *  
     */
    public void complexInverse(final float[] a, final boolean scale)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            columns = 2 * columns;
            if ((nthreads > 1) && useThreads) {
                xdft2d0_subth1(0, 1, a, scale);
                cdft2d_subth(1, a, scale);
            } else {

                for (int r = 0; r < rows; r++) {
                    fftColumns.complexInverse(a, r * columns, scale);
                }
                cdft2d_sub(1, a, scale);
            }
            columns = columns / 2;
        } else {
            final int rowspan = 2 * columns;
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
                            for (int r = firstRow; r < lastRow; r++) {
                                fftColumns.complexInverse(a, r * rowspan, scale);
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
                }
                p = columns / nthreads;
                for (int l = 0; l < nthreads; l++) {
                    final int firstColumn = l * p;
                    final int lastColumn = (l == (nthreads - 1)) ? columns : firstColumn + p;
                    futures[l] = ConcurrencyUtils.submit(new Runnable()
                    {
                        public void run()
                        {
                            float[] temp = new float[2 * rows];
                            for (int c = firstColumn; c < lastColumn; c++) {
                                int idx1 = 2 * c;
                                for (int r = 0; r < rows; r++) {
                                    int idx2 = 2 * r;
                                    int idx3 = r * rowspan + idx1;
                                    temp[idx2] = a[idx3];
                                    temp[idx2 + 1] = a[idx3 + 1];
                                }
                                fftRows.complexInverse(temp, scale);
                                for (int r = 0; r < rows; r++) {
                                    int idx2 = 2 * r;
                                    int idx3 = r * rowspan + idx1;
                                    a[idx3] = temp[idx2];
                                    a[idx3 + 1] = temp[idx2 + 1];
                                }
                            }
                        }
                    });
                }
                try {
                    ConcurrencyUtils.waitForCompletion(futures);
                } catch (InterruptedException ex) {
                    Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
                } catch (ExecutionException ex) {
                    Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else {
                for (int r = 0; r < rows; r++) {
                    fftColumns.complexInverse(a, r * rowspan, scale);
                }
                float[] temp = new float[2 * rows];
                for (int c = 0; c < columns; c++) {
                    int idx1 = 2 * c;
                    for (int r = 0; r < rows; r++) {
                        int idx2 = 2 * r;
                        int idx3 = r * rowspan + idx1;
                        temp[idx2] = a[idx3];
                        temp[idx2 + 1] = a[idx3 + 1];
                    }
                    fftRows.complexInverse(temp, scale);
                    for (int r = 0; r < rows; r++) {
                        int idx2 = 2 * r;
                        int idx3 = r * rowspan + idx1;
                        a[idx3] = temp[idx2];
                        a[idx3 + 1] = temp[idx2 + 1];
                    }
                }
            }
        }
    }

    /**
     * Computes 2D inverse DFT of complex data leaving the result in
     * <code>a</code>. The data is stored in 1D array in row-major order.
     * Complex number is stored as two float values in sequence: the real and
     * imaginary part, i.e. the input array must be of size rows*2*columns. The
     * physical layout of the input data has to be as follows:<br>
     *  
     * <pre>
     * a[k1*2*columns+2*k2] = Re[k1][k2],
     * a[k1*2*columns+2*k2+1] = Im[k1][k2], 0&lt;=k1&lt;rows, 0&lt;=k2&lt;columns,
     * </pre>
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     *  
     */
    public void complexInverse(final FloatLargeArray a, final boolean scale)
    {
        if (!a.isLarge() && !a.isConstant()) {
            complexInverse(a.getData(), scale);
        } else {
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if (isPowerOfTwo) {
                columnsl = 2 * columnsl;
                if ((nthreads > 1) && useThreads) {
                    xdft2d0_subth1(0, 1, a, scale);
                    cdft2d_subth(1, a, scale);
                } else {

                    for (long r = 0; r < rowsl; r++) {
                        fftColumns.complexInverse(a, r * columnsl, scale);
                    }
                    cdft2d_sub(1, a, scale);
                }
                columnsl = columnsl / 2;
            } else {
                final long rowspan = 2 * columnsl;
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
                                for (long r = firstRow; r < lastRow; r++) {
                                    fftColumns.complexInverse(a, r * rowspan, scale);
                                }
                            }
                        });
                    }
                    try {
                        ConcurrencyUtils.waitForCompletion(futures);
                    } catch (InterruptedException ex) {
                        Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
                    } catch (ExecutionException ex) {
                        Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
                    }
                    p = columnsl / nthreads;
                    for (int l = 0; l < nthreads; l++) {
                        final long firstColumn = l * p;
                        final long lastColumn = (l == (nthreads - 1)) ? columnsl : firstColumn + p;
                        futures[l] = ConcurrencyUtils.submit(new Runnable()
                        {
                            public void run()
                            {
                                FloatLargeArray temp = new FloatLargeArray(2 * rowsl, false);
                                for (long c = firstColumn; c < lastColumn; c++) {
                                    long idx1 = 2 * c;
                                    for (long r = 0; r < rowsl; r++) {
                                        long idx2 = 2 * r;
                                        long idx3 = r * rowspan + idx1;
                                        temp.setDouble(idx2, a.getFloat(idx3));
                                        temp.setDouble(idx2 + 1, a.getFloat(idx3 + 1));
                                    }
                                    fftRows.complexInverse(temp, scale);
                                    for (long r = 0; r < rowsl; r++) {
                                        long idx2 = 2 * r;
                                        long idx3 = r * rowspan + idx1;
                                        a.setDouble(idx3, temp.getFloat(idx2));
                                        a.setDouble(idx3 + 1, temp.getFloat(idx2 + 1));
                                    }
                                }
                            }
                        });
                    }
                    try {
                        ConcurrencyUtils.waitForCompletion(futures);
                    } catch (InterruptedException ex) {
                        Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
                    } catch (ExecutionException ex) {
                        Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
                    }
                } else {
                    for (long r = 0; r < rowsl; r++) {
                        fftColumns.complexInverse(a, r * rowspan, scale);
                    }
                    FloatLargeArray temp = new FloatLargeArray(2 * rowsl);
                    for (long c = 0; c < columnsl; c++) {
                        long idx1 = 2 * c;
                        for (long r = 0; r < rowsl; r++) {
                            long idx2 = 2 * r;
                            long idx3 = r * rowspan + idx1;
                            temp.setDouble(idx2, a.getFloat(idx3));
                            temp.setDouble(idx2 + 1, a.getFloat(idx3 + 1));
                        }
                        fftRows.complexInverse(temp, scale);
                        for (long r = 0; r < rowsl; r++) {
                            long idx2 = 2 * r;
                            long idx3 = r * rowspan + idx1;
                            a.setDouble(idx3, temp.getFloat(idx2));
                            a.setDouble(idx3 + 1, temp.getFloat(idx2 + 1));
                        }
                    }
                }
            }
        }
    }

    /**
     * Computes 2D inverse DFT of complex data leaving the result in
     * <code>a</code>. The data is stored in 2D array. Complex data is
     * represented by 2 float values in sequence: the real and imaginary part,
     * i.e. the input array must be of size rows by 2*columns. The physical
     * layout of the input data has to be as follows:<br>
     *  
     * <pre>
     * a[k1][2*k2] = Re[k1][k2],
     * a[k1][2*k2+1] = Im[k1][k2], 0&lt;=k1&lt;rows, 0&lt;=k2&lt;columns,
     * </pre>
     *  
     * @param a     data to transform
     * @param scale if true then scaling is performed
     *  
     */
    public void complexInverse(final float[][] a, final boolean scale)
    {
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (isPowerOfTwo) {
            columns = 2 * columns;
            if ((nthreads > 1) && useThreads) {
                xdft2d0_subth1(0, 1, a, scale);
                cdft2d_subth(1, a, scale);
            } else {

                for (int r = 0; r < rows; r++) {
                    fftColumns.complexInverse(a[r], scale);
                }
                cdft2d_sub(1, a, scale);
            }
            columns = columns / 2;
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
                            fftColumns.complexInverse(a[r], scale);
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }
            p = columns / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstColumn = l * p;
                final int lastColumn = (l == (nthreads - 1)) ? columns : firstColumn + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        float[] temp = new float[2 * rows];
                        for (int c = firstColumn; c < lastColumn; c++) {
                            int idx1 = 2 * c;
                            for (int r = 0; r < rows; r++) {
                                int idx2 = 2 * r;
                                temp[idx2] = a[r][idx1];
                                temp[idx2 + 1] = a[r][idx1 + 1];
                            }
                            fftRows.complexInverse(temp, scale);
                            for (int r = 0; r < rows; r++) {
                                int idx2 = 2 * r;
                                a[r][idx1] = temp[idx2];
                                a[r][idx1 + 1] = temp[idx2 + 1];
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {
            for (int r = 0; r < rows; r++) {
                fftColumns.complexInverse(a[r], scale);
            }
            float[] temp = new float[2 * rows];
            for (int c = 0; c < columns; c++) {
                int idx1 = 2 * c;
                for (int r = 0; r < rows; r++) {
                    int idx2 = 2 * r;
                    temp[idx2] = a[r][idx1];
                    temp[idx2 + 1] = a[r][idx1 + 1];
                }
                fftRows.complexInverse(temp, scale);
                for (int r = 0; r < rows; r++) {
                    int idx2 = 2 * r;
                    a[r][idx1] = temp[idx2];
                    a[r][idx1 + 1] = temp[idx2 + 1];
                }
            }
        }
    }

    /**
     * Computes 2D forward DFT of real data leaving the result in <code>a</code>
     * . This method only works when the sizes of both dimensions are
     * power-of-two numbers. The physical layout of the output data is as
     * follows:
     *  
     * <pre>
     * a[k1*columns+2*k2] = Re[k1][k2] = Re[rows-k1][columns-k2],
     * a[k1*columns+2*k2+1] = Im[k1][k2] = -Im[rows-k1][columns-k2],
     *       0&lt;k1&lt;rows, 0&lt;k2&lt;columns/2,
     * a[2*k2] = Re[0][k2] = Re[0][columns-k2],
     * a[2*k2+1] = Im[0][k2] = -Im[0][columns-k2],
     *       0&lt;k2&lt;columns/2,
     * a[k1*columns] = Re[k1][0] = Re[rows-k1][0],
     * a[k1*columns+1] = Im[k1][0] = -Im[rows-k1][0],
     * a[(rows-k1)*columns+1] = Re[k1][columns/2] = Re[rows-k1][columns/2],
     * a[(rows-k1)*columns] = -Im[k1][columns/2] = Im[rows-k1][columns/2],
     *       0&lt;k1&lt;rows/2,
     * a[0] = Re[0][0],
     * a[1] = Re[0][columns/2],
     * a[(rows/2)*columns] = Re[rows/2][0],
     * a[(rows/2)*columns+1] = Re[rows/2][columns/2]
     * </pre>
     *  
     * This method computes only half of the elements of the real transform. The
     * other half satisfies the symmetry condition. If you want the full real
     * forward transform, use <code>realForwardFull</code>. To get back the
     * original data, use <code>realInverse</code> on the output of this method.
     *  
     * @param a data to transform
     */
    public void realForward(float[] a)
    {
        if (isPowerOfTwo == false) {
            throw new IllegalArgumentException("rows and columns must be power of two numbers");
        } else {
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && useThreads) {
                xdft2d0_subth1(1, 1, a, true);
                cdft2d_subth(-1, a, true);
                rdft2d_sub(1, a);
            } else {
                for (int r = 0; r < rows; r++) {
                    fftColumns.realForward(a, r * columns);
                }
                cdft2d_sub(-1, a, true);
                rdft2d_sub(1, a);
            }
        }
    }

    /**
     * Computes 2D forward DFT of real data leaving the result in <code>a</code>
     * . This method only works when the sizes of both dimensions are
     * power-of-two numbers. The physical layout of the output data is as
     * follows:
     *  
     * <pre>
     * a[k1*columns+2*k2] = Re[k1][k2] = Re[rows-k1][columns-k2],
     * a[k1*columns+2*k2+1] = Im[k1][k2] = -Im[rows-k1][columns-k2],
     *       0&lt;k1&lt;rows, 0&lt;k2&lt;columns/2,
     * a[2*k2] = Re[0][k2] = Re[0][columns-k2],
     * a[2*k2+1] = Im[0][k2] = -Im[0][columns-k2],
     *       0&lt;k2&lt;columns/2,
     * a[k1*columns] = Re[k1][0] = Re[rows-k1][0],
     * a[k1*columns+1] = Im[k1][0] = -Im[rows-k1][0],
     * a[(rows-k1)*columns+1] = Re[k1][columns/2] = Re[rows-k1][columns/2],
     * a[(rows-k1)*columns] = -Im[k1][columns/2] = Im[rows-k1][columns/2],
     *       0&lt;k1&lt;rows/2,
     * a[0] = Re[0][0],
     * a[1] = Re[0][columns/2],
     * a[(rows/2)*columns] = Re[rows/2][0],
     * a[(rows/2)*columns+1] = Re[rows/2][columns/2]
     * </pre>
     *  
     * This method computes only half of the elements of the real transform. The
     * other half satisfies the symmetry condition. If you want the full real
     * forward transform, use <code>realForwardFull</code>. To get back the
     * original data, use <code>realInverse</code> on the output of this method.
     *  
     * @param a data to transform
     */
    public void realForward(FloatLargeArray a)
    {
        if (isPowerOfTwo == false) {
            throw new IllegalArgumentException("rows and columns must be power of two numbers");
        } else {
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && useThreads) {
                xdft2d0_subth1(1, 1, a, true);
                cdft2d_subth(-1, a, true);
                rdft2d_sub(1, a);
            } else {
                for (long r = 0; r < rowsl; r++) {
                    fftColumns.realForward(a, r * columnsl);
                }
                cdft2d_sub(-1, a, true);
                rdft2d_sub(1, a);
            }
        }
    }

    /**
     * Computes 2D forward DFT of real data leaving the result in <code>a</code>
     * . This method only works when the sizes of both dimensions are
     * power-of-two numbers. The physical layout of the output data is as
     * follows:
     *  
     * <pre>
     * a[k1][2*k2] = Re[k1][k2] = Re[rows-k1][columns-k2],
     * a[k1][2*k2+1] = Im[k1][k2] = -Im[rows-k1][columns-k2],
     *       0&lt;k1&lt;rows, 0&lt;k2&lt;columns/2,
     * a[0][2*k2] = Re[0][k2] = Re[0][columns-k2],
     * a[0][2*k2+1] = Im[0][k2] = -Im[0][columns-k2],
     *       0&lt;k2&lt;columns/2,
     * a[k1][0] = Re[k1][0] = Re[rows-k1][0],
     * a[k1][1] = Im[k1][0] = -Im[rows-k1][0],
     * a[rows-k1][1] = Re[k1][columns/2] = Re[rows-k1][columns/2],
     * a[rows-k1][0] = -Im[k1][columns/2] = Im[rows-k1][columns/2],
     *       0&lt;k1&lt;rows/2,
     * a[0][0] = Re[0][0],
     * a[0][1] = Re[0][columns/2],
     * a[rows/2][0] = Re[rows/2][0],
     * a[rows/2][1] = Re[rows/2][columns/2]
     * </pre>
     *  
     * This method computes only half of the elements of the real transform. The
     * other half satisfies the symmetry condition. If you want the full real
     * forward transform, use <code>realForwardFull</code>. To get back the
     * original data, use <code>realInverse</code> on the output of this method.
     *  
     * @param a data to transform
     */
    public void realForward(float[][] a)
    {
        if (isPowerOfTwo == false) {
            throw new IllegalArgumentException("rows and columns must be power of two numbers");
        } else {
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && useThreads) {
                xdft2d0_subth1(1, 1, a, true);
                cdft2d_subth(-1, a, true);
                rdft2d_sub(1, a);
            } else {
                for (int r = 0; r < rows; r++) {
                    fftColumns.realForward(a[r]);
                }
                cdft2d_sub(-1, a, true);
                rdft2d_sub(1, a);
            }
        }
    }

    /**
     * Computes 2D forward DFT of real data leaving the result in <code>a</code>
     * . This method computes full real forward transform, i.e. you will get the
     * same result as from <code>complexForward</code> called with all imaginary
     * part equal 0. Because the result is stored in <code>a</code>, the input
     * array must be of size rows*2*columns, with only the first rows*columns
     * elements filled with real data. To get back the original data, use
     * <code>complexInverse</code> on the output of this method.
     *  
     * @param a data to transform
     */
    public void realForwardFull(float[] a)
    {
        if (isPowerOfTwo) {
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && useThreads) {
                xdft2d0_subth1(1, 1, a, true);
                cdft2d_subth(-1, a, true);
                rdft2d_sub(1, a);
            } else {
                for (int r = 0; r < rows; r++) {
                    fftColumns.realForward(a, r * columns);
                }
                cdft2d_sub(-1, a, true);
                rdft2d_sub(1, a);
            }
            fillSymmetric(a);
        } else {
            mixedRadixRealForwardFull(a);
        }
    }

    /**
     * Computes 2D forward DFT of real data leaving the result in <code>a</code>
     * . This method computes full real forward transform, i.e. you will get the
     * same result as from <code>complexForward</code> called with all imaginary
     * part equal 0. Because the result is stored in <code>a</code>, the input
     * array must be of size rows*2*columns, with only the first rows*columns
     * elements filled with real data. To get back the original data, use
     * <code>complexInverse</code> on the output of this method.
     *  
     * @param a data to transform
     */
    public void realForwardFull(FloatLargeArray a)
    {
        if (isPowerOfTwo) {
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && useThreads) {
                xdft2d0_subth1(1, 1, a, true);
                cdft2d_subth(-1, a, true);
                rdft2d_sub(1, a);
            } else {
                for (long r = 0; r < rowsl; r++) {
                    fftColumns.realForward(a, r * columnsl);
                }
                cdft2d_sub(-1, a, true);
                rdft2d_sub(1, a);
            }
            fillSymmetric(a);
        } else {
            mixedRadixRealForwardFull(a);
        }
    }

    /**
     * Computes 2D forward DFT of real data leaving the result in <code>a</code>
     * . This method computes full real forward transform, i.e. you will get the
     * same result as from <code>complexForward</code> called with all imaginary
     * part equal 0. Because the result is stored in <code>a</code>, the input
     * array must be of size rows by 2*columns, with only the first rows by
     * columns elements filled with real data. To get back the original data,
     * use <code>complexInverse</code> on the output of this method.
     *  
     * @param a data to transform
     */
    public void realForwardFull(float[][] a)
    {
        if (isPowerOfTwo) {
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && useThreads) {
                xdft2d0_subth1(1, 1, a, true);
                cdft2d_subth(-1, a, true);
                rdft2d_sub(1, a);
            } else {
                for (int r = 0; r < rows; r++) {
                    fftColumns.realForward(a[r]);
                }
                cdft2d_sub(-1, a, true);
                rdft2d_sub(1, a);
            }
            fillSymmetric(a);
        } else {
            mixedRadixRealForwardFull(a);
        }
    }

    /**
     * Computes 2D inverse DFT of real data leaving the result in <code>a</code>
     * . This method only works when the sizes of both dimensions are
     * power-of-two numbers. The physical layout of the input data has to be as
     * follows:
     *  
     * <pre>
     * a[k1*columns+2*k2] = Re[k1][k2] = Re[rows-k1][columns-k2],
     * a[k1*columns+2*k2+1] = Im[k1][k2] = -Im[rows-k1][columns-k2],
     *       0&lt;k1&lt;rows, 0&lt;k2&lt;columns/2,
     * a[2*k2] = Re[0][k2] = Re[0][columns-k2],
     * a[2*k2+1] = Im[0][k2] = -Im[0][columns-k2],
     *       0&lt;k2&lt;columns/2,
     * a[k1*columns] = Re[k1][0] = Re[rows-k1][0],
     * a[k1*columns+1] = Im[k1][0] = -Im[rows-k1][0],
     * a[(rows-k1)*columns+1] = Re[k1][columns/2] = Re[rows-k1][columns/2],
     * a[(rows-k1)*columns] = -Im[k1][columns/2] = Im[rows-k1][columns/2],
     *       0&lt;k1&lt;rows/2,
     * a[0] = Re[0][0],
     * a[1] = Re[0][columns/2],
     * a[(rows/2)*columns] = Re[rows/2][0],
     * a[(rows/2)*columns+1] = Re[rows/2][columns/2]
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
    public void realInverse(float[] a, boolean scale)
    {
        if (isPowerOfTwo == false) {
            throw new IllegalArgumentException("rows and columns must be power of two numbers");
        } else {
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && useThreads) {
                rdft2d_sub(-1, a);
                cdft2d_subth(1, a, scale);
                xdft2d0_subth1(1, -1, a, scale);
            } else {
                rdft2d_sub(-1, a);
                cdft2d_sub(1, a, scale);
                for (int r = 0; r < rows; r++) {
                    fftColumns.realInverse(a, r * columns, scale);
                }
            }
        }
    }

    /**
     * Computes 2D inverse DFT of real data leaving the result in <code>a</code>
     * . This method only works when the sizes of both dimensions are
     * power-of-two numbers. The physical layout of the input data has to be as
     * follows:
     *  
     * <pre>
     * a[k1*columns+2*k2] = Re[k1][k2] = Re[rows-k1][columns-k2],
     * a[k1*columns+2*k2+1] = Im[k1][k2] = -Im[rows-k1][columns-k2],
     *       0&lt;k1&lt;rows, 0&lt;k2&lt;columns/2,
     * a[2*k2] = Re[0][k2] = Re[0][columns-k2],
     * a[2*k2+1] = Im[0][k2] = -Im[0][columns-k2],
     *       0&lt;k2&lt;columns/2,
     * a[k1*columns] = Re[k1][0] = Re[rows-k1][0],
     * a[k1*columns+1] = Im[k1][0] = -Im[rows-k1][0],
     * a[(rows-k1)*columns+1] = Re[k1][columns/2] = Re[rows-k1][columns/2],
     * a[(rows-k1)*columns] = -Im[k1][columns/2] = Im[rows-k1][columns/2],
     *       0&lt;k1&lt;rows/2,
     * a[0] = Re[0][0],
     * a[1] = Re[0][columns/2],
     * a[(rows/2)*columns] = Re[rows/2][0],
     * a[(rows/2)*columns+1] = Re[rows/2][columns/2]
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
    public void realInverse(FloatLargeArray a, boolean scale)
    {
        if (isPowerOfTwo == false) {
            throw new IllegalArgumentException("rows and columns must be power of two numbers");
        } else {
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && useThreads) {
                rdft2d_sub(-1, a);
                cdft2d_subth(1, a, scale);
                xdft2d0_subth1(1, -1, a, scale);
            } else {
                rdft2d_sub(-1, a);
                cdft2d_sub(1, a, scale);
                for (long r = 0; r < rowsl; r++) {
                    fftColumns.realInverse(a, r * columnsl, scale);
                }
            }
        }
    }

    /**
     * Computes 2D inverse DFT of real data leaving the result in <code>a</code>
     * . This method only works when the sizes of both dimensions are
     * power-of-two numbers. The physical layout of the input data has to be as
     * follows:
     *  
     * <pre>
     * a[k1][2*k2] = Re[k1][k2] = Re[rows-k1][columns-k2],
     * a[k1][2*k2+1] = Im[k1][k2] = -Im[rows-k1][columns-k2],
     *       0&lt;k1&lt;rows, 0&lt;k2&lt;columns/2,
     * a[0][2*k2] = Re[0][k2] = Re[0][columns-k2],
     * a[0][2*k2+1] = Im[0][k2] = -Im[0][columns-k2],
     *       0&lt;k2&lt;columns/2,
     * a[k1][0] = Re[k1][0] = Re[rows-k1][0],
     * a[k1][1] = Im[k1][0] = -Im[rows-k1][0],
     * a[rows-k1][1] = Re[k1][columns/2] = Re[rows-k1][columns/2],
     * a[rows-k1][0] = -Im[k1][columns/2] = Im[rows-k1][columns/2],
     *       0&lt;k1&lt;rows/2,
     * a[0][0] = Re[0][0],
     * a[0][1] = Re[0][columns/2],
     * a[rows/2][0] = Re[rows/2][0],
     * a[rows/2][1] = Re[rows/2][columns/2]
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
    public void realInverse(float[][] a, boolean scale)
    {
        if (isPowerOfTwo == false) {
            throw new IllegalArgumentException("rows and columns must be power of two numbers");
        } else {
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && useThreads) {
                rdft2d_sub(-1, a);
                cdft2d_subth(1, a, scale);
                xdft2d0_subth1(1, -1, a, scale);
            } else {
                rdft2d_sub(-1, a);
                cdft2d_sub(1, a, scale);
                for (int r = 0; r < rows; r++) {
                    fftColumns.realInverse(a[r], scale);
                }
            }
        }
    }

    /**
     * Computes 2D inverse DFT of real data leaving the result in <code>a</code>
     * . This method computes full real inverse transform, i.e. you will get the
     * same result as from <code>complexInverse</code> called with all imaginary
     * part equal 0. Because the result is stored in <code>a</code>, the input
     * array must be of size rows*2*columns, with only the first rows*columns
     * elements filled with real data.
     *  
     * @param a     data to transform
     *  
     * @param scale if true then scaling is performed
     */
    public void realInverseFull(float[] a, boolean scale)
    {
        if (isPowerOfTwo) {
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && useThreads) {
                xdft2d0_subth2(1, -1, a, scale);
                cdft2d_subth(1, a, scale);
                rdft2d_sub(1, a);
            } else {
                for (int r = 0; r < rows; r++) {
                    fftColumns.realInverse2(a, r * columns, scale);
                }
                cdft2d_sub(1, a, scale);
                rdft2d_sub(1, a);
            }
            fillSymmetric(a);
        } else {
            mixedRadixRealInverseFull(a, scale);
        }
    }

    /**
     * Computes 2D inverse DFT of real data leaving the result in <code>a</code>
     * . This method computes full real inverse transform, i.e. you will get the
     * same result as from <code>complexInverse</code> called with all imaginary
     * part equal 0. Because the result is stored in <code>a</code>, the input
     * array must be of size rows*2*columns, with only the first rows*columns
     * elements filled with real data.
     *  
     * @param a     data to transform
     *  
     * @param scale if true then scaling is performed
     */
    public void realInverseFull(FloatLargeArray a, boolean scale)
    {
        if (isPowerOfTwo) {
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && useThreads) {
                xdft2d0_subth2(1, -1, a, scale);
                cdft2d_subth(1, a, scale);
                rdft2d_sub(1, a);
            } else {
                for (long r = 0; r < rowsl; r++) {
                    fftColumns.realInverse2(a, r * columnsl, scale);
                }
                cdft2d_sub(1, a, scale);
                rdft2d_sub(1, a);
            }
            fillSymmetric(a);
        } else {
            mixedRadixRealInverseFull(a, scale);
        }
    }

    /**
     * Computes 2D inverse DFT of real data leaving the result in <code>a</code>
     * . This method computes full real inverse transform, i.e. you will get the
     * same result as from <code>complexInverse</code> called with all imaginary
     * part equal 0. Because the result is stored in <code>a</code>, the input
     * array must be of size rows by 2*columns, with only the first rows by
     * columns elements filled with real data.
     *  
     * @param a     data to transform
     *  
     * @param scale if true then scaling is performed
     */
    public void realInverseFull(float[][] a, boolean scale)
    {
        if (isPowerOfTwo) {
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && useThreads) {
                xdft2d0_subth2(1, -1, a, scale);
                cdft2d_subth(1, a, scale);
                rdft2d_sub(1, a);
            } else {
                for (int r = 0; r < rows; r++) {
                    fftColumns.realInverse2(a[r], 0, scale);
                }
                cdft2d_sub(1, a, scale);
                rdft2d_sub(1, a);
            }
            fillSymmetric(a);
        } else {
            mixedRadixRealInverseFull(a, scale);
        }
    }

    private void mixedRadixRealForwardFull(final float[][] a)
    {
        final int n2d2 = columns / 2 + 1;
        final float[][] temp = new float[n2d2][2 * rows];

        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && useThreads && (rows >= nthreads) && (n2d2 - 2 >= nthreads)) {
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
                            fftColumns.realForward(a[i]);
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int r = 0; r < rows; r++) {
                temp[0][r] = a[r][0]; //first column is always real
            }
            fftRows.realForwardFull(temp[0]);

            p = (n2d2 - 2) / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstColumn = 1 + l * p;
                final int lastColumn = (l == (nthreads - 1)) ? n2d2 - 1 : firstColumn + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (int c = firstColumn; c < lastColumn; c++) {
                            int idx2 = 2 * c;
                            for (int r = 0; r < rows; r++) {
                                int idx1 = 2 * r;
                                temp[c][idx1] = a[r][idx2];
                                temp[c][idx1 + 1] = a[r][idx2 + 1];
                            }
                            fftRows.complexForward(temp[c]);
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }

            if ((columns % 2) == 0) {
                for (int r = 0; r < rows; r++) {
                    temp[n2d2 - 1][r] = a[r][1];
                    //imaginary part = 0;
                }
                fftRows.realForwardFull(temp[n2d2 - 1]);

            } else {
                for (int r = 0; r < rows; r++) {
                    int idx1 = 2 * r;
                    int idx2 = n2d2 - 1;
                    temp[idx2][idx1] = a[r][2 * idx2];
                    temp[idx2][idx1 + 1] = a[r][1];
                }
                fftRows.complexForward(temp[n2d2 - 1]);

            }

            p = rows / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstRow = l * p;
                final int lastRow = (l == (nthreads - 1)) ? rows : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (int r = firstRow; r < lastRow; r++) {
                            int idx1 = 2 * r;
                            for (int c = 0; c < n2d2; c++) {
                                int idx2 = 2 * c;
                                a[r][idx2] = temp[c][idx1];
                                a[r][idx2 + 1] = temp[c][idx1 + 1];
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int l = 0; l < nthreads; l++) {
                final int firstRow = 1 + l * p;
                final int lastRow = (l == (nthreads - 1)) ? rows : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (int r = firstRow; r < lastRow; r++) {
                            int idx3 = rows - r;
                            for (int c = n2d2; c < columns; c++) {
                                int idx1 = 2 * c;
                                int idx2 = 2 * (columns - c);
                                a[0][idx1] = a[0][idx2];
                                a[0][idx1 + 1] = -a[0][idx2 + 1];
                                a[r][idx1] = a[idx3][idx2];
                                a[r][idx1 + 1] = -a[idx3][idx2 + 1];
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }

        } else {
            for (int r = 0; r < rows; r++) {
                fftColumns.realForward(a[r]);
            }

            for (int r = 0; r < rows; r++) {
                temp[0][r] = a[r][0]; //first column is always real
            }
            fftRows.realForwardFull(temp[0]);

            for (int c = 1; c < n2d2 - 1; c++) {
                int idx2 = 2 * c;
                for (int r = 0; r < rows; r++) {
                    int idx1 = 2 * r;
                    temp[c][idx1] = a[r][idx2];
                    temp[c][idx1 + 1] = a[r][idx2 + 1];
                }
                fftRows.complexForward(temp[c]);
            }

            if ((columns % 2) == 0) {
                for (int r = 0; r < rows; r++) {
                    temp[n2d2 - 1][r] = a[r][1];
                    //imaginary part = 0;
                }
                fftRows.realForwardFull(temp[n2d2 - 1]);

            } else {
                for (int r = 0; r < rows; r++) {
                    int idx1 = 2 * r;
                    int idx2 = n2d2 - 1;
                    temp[idx2][idx1] = a[r][2 * idx2];
                    temp[idx2][idx1 + 1] = a[r][1];
                }
                fftRows.complexForward(temp[n2d2 - 1]);

            }

            for (int r = 0; r < rows; r++) {
                int idx1 = 2 * r;
                for (int c = 0; c < n2d2; c++) {
                    int idx2 = 2 * c;
                    a[r][idx2] = temp[c][idx1];
                    a[r][idx2 + 1] = temp[c][idx1 + 1];
                }
            }

            //fill symmetric
            for (int r = 1; r < rows; r++) {
                int idx3 = rows - r;
                for (int c = n2d2; c < columns; c++) {
                    int idx1 = 2 * c;
                    int idx2 = 2 * (columns - c);
                    a[0][idx1] = a[0][idx2];
                    a[0][idx1 + 1] = -a[0][idx2 + 1];
                    a[r][idx1] = a[idx3][idx2];
                    a[r][idx1 + 1] = -a[idx3][idx2 + 1];
                }
            }
        }
    }

    private void mixedRadixRealForwardFull(final float[] a)
    {
        final int rowStride = 2 * columns;
        final int n2d2 = columns / 2 + 1;
        final float[][] temp = new float[n2d2][2 * rows];

        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && useThreads && (rows >= nthreads) && (n2d2 - 2 >= nthreads)) {
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
                            fftColumns.realForward(a, i * columns);
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int r = 0; r < rows; r++) {
                temp[0][r] = a[r * columns]; //first column is always real
            }
            fftRows.realForwardFull(temp[0]);

            p = (n2d2 - 2) / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstColumn = 1 + l * p;
                final int lastColumn = (l == (nthreads - 1)) ? n2d2 - 1 : firstColumn + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (int c = firstColumn; c < lastColumn; c++) {
                            int idx0 = 2 * c;
                            for (int r = 0; r < rows; r++) {
                                int idx1 = 2 * r;
                                int idx2 = r * columns + idx0;
                                temp[c][idx1] = a[idx2];
                                temp[c][idx1 + 1] = a[idx2 + 1];
                            }
                            fftRows.complexForward(temp[c]);
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }

            if ((columns % 2) == 0) {
                for (int r = 0; r < rows; r++) {
                    temp[n2d2 - 1][r] = a[r * columns + 1];
                    //imaginary part = 0;
                }
                fftRows.realForwardFull(temp[n2d2 - 1]);

            } else {
                for (int r = 0; r < rows; r++) {
                    int idx1 = 2 * r;
                    int idx2 = r * columns;
                    int idx3 = n2d2 - 1;
                    temp[idx3][idx1] = a[idx2 + 2 * idx3];
                    temp[idx3][idx1 + 1] = a[idx2 + 1];
                }
                fftRows.complexForward(temp[n2d2 - 1]);
            }

            p = rows / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstRow = l * p;
                final int lastRow = (l == (nthreads - 1)) ? rows : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (int r = firstRow; r < lastRow; r++) {
                            int idx1 = 2 * r;
                            for (int c = 0; c < n2d2; c++) {
                                int idx0 = 2 * c;
                                int idx2 = r * rowStride + idx0;
                                a[idx2] = temp[c][idx1];
                                a[idx2 + 1] = temp[c][idx1 + 1];
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int l = 0; l < nthreads; l++) {
                final int firstRow = 1 + l * p;
                final int lastRow = (l == (nthreads - 1)) ? rows : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (int r = firstRow; r < lastRow; r++) {
                            int idx5 = r * rowStride;
                            int idx6 = (rows - r + 1) * rowStride;
                            for (int c = n2d2; c < columns; c++) {
                                int idx1 = 2 * c;
                                int idx2 = 2 * (columns - c);
                                a[idx1] = a[idx2];
                                a[idx1 + 1] = -a[idx2 + 1];
                                int idx3 = idx5 + idx1;
                                int idx4 = idx6 - idx1;
                                a[idx3] = a[idx4];
                                a[idx3 + 1] = -a[idx4 + 1];
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }

        } else {
            for (int r = 0; r < rows; r++) {
                fftColumns.realForward(a, r * columns);
            }
            for (int r = 0; r < rows; r++) {
                temp[0][r] = a[r * columns]; //first column is always real
            }
            fftRows.realForwardFull(temp[0]);

            for (int c = 1; c < n2d2 - 1; c++) {
                int idx0 = 2 * c;
                for (int r = 0; r < rows; r++) {
                    int idx1 = 2 * r;
                    int idx2 = r * columns + idx0;
                    temp[c][idx1] = a[idx2];
                    temp[c][idx1 + 1] = a[idx2 + 1];
                }
                fftRows.complexForward(temp[c]);
            }

            if ((columns % 2) == 0) {
                for (int r = 0; r < rows; r++) {
                    temp[n2d2 - 1][r] = a[r * columns + 1];
                    //imaginary part = 0;
                }
                fftRows.realForwardFull(temp[n2d2 - 1]);

            } else {
                for (int r = 0; r < rows; r++) {
                    int idx1 = 2 * r;
                    int idx2 = r * columns;
                    int idx3 = n2d2 - 1;
                    temp[idx3][idx1] = a[idx2 + 2 * idx3];
                    temp[idx3][idx1 + 1] = a[idx2 + 1];
                }
                fftRows.complexForward(temp[n2d2 - 1]);
            }

            for (int r = 0; r < rows; r++) {
                int idx1 = 2 * r;
                for (int c = 0; c < n2d2; c++) {
                    int idx0 = 2 * c;
                    int idx2 = r * rowStride + idx0;
                    a[idx2] = temp[c][idx1];
                    a[idx2 + 1] = temp[c][idx1 + 1];
                }
            }

            //fill symmetric
            for (int r = 1; r < rows; r++) {
                int idx5 = r * rowStride;
                int idx6 = (rows - r + 1) * rowStride;
                for (int c = n2d2; c < columns; c++) {
                    int idx1 = 2 * c;
                    int idx2 = 2 * (columns - c);
                    a[idx1] = a[idx2];
                    a[idx1 + 1] = -a[idx2 + 1];
                    int idx3 = idx5 + idx1;
                    int idx4 = idx6 - idx1;
                    a[idx3] = a[idx4];
                    a[idx3 + 1] = -a[idx4 + 1];
                }
            }
        }
    }

    private void mixedRadixRealForwardFull(final FloatLargeArray a)
    {
        final long rowStride = 2 * columnsl;
        final long n2d2 = columnsl / 2 + 1;
        final FloatLargeArray temp = new FloatLargeArray(n2d2 * 2 * rowsl);
        final long temp_stride = 2 * rowsl;

        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && useThreads && (rowsl >= nthreads) && (n2d2 - 2 >= nthreads)) {
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
                            fftColumns.realForward(a, i * columnsl);
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (long r = 0; r < rowsl; r++) {
                temp.setDouble(r, a.getFloat(r * columnsl)); //first column is always real
            }
            fftRows.realForwardFull(temp);

            p = (n2d2 - 2) / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final long firstColumn = 1 + l * p;
                final long lastColumn = (l == (nthreads - 1)) ? n2d2 - 1 : firstColumn + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (long c = firstColumn; c < lastColumn; c++) {
                            long idx0 = 2 * c;
                            for (long r = 0; r < rowsl; r++) {
                                long idx1 = 2 * r;
                                long idx2 = r * columnsl + idx0;
                                temp.setDouble(c * temp_stride + idx1, a.getFloat(idx2));
                                temp.setDouble(c * temp_stride + idx1 + 1, a.getFloat(idx2 + 1));
                            }
                            fftRows.complexForward(temp, c * temp_stride);
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }

            if ((columnsl % 2) == 0) {
                for (long r = 0; r < rowsl; r++) {
                    temp.setDouble((n2d2 - 1) * temp_stride + r, a.getFloat(r * columnsl + 1));
                    //imaginary part = 0;
                }
                fftRows.realForwardFull(temp, (n2d2 - 1) * temp_stride);

            } else {
                for (long r = 0; r < rowsl; r++) {
                    long idx1 = 2 * r;
                    long idx2 = r * columnsl;
                    long idx3 = n2d2 - 1;
                    temp.setDouble(idx3 * temp_stride + idx1, a.getFloat(idx2 + 2 * idx3));
                    temp.setDouble(idx3 * temp_stride + idx1 + 1, a.getFloat(idx2 + 1));
                }
                fftRows.complexForward(temp, (n2d2 - 1) * temp_stride);
            }

            p = rowsl / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final long firstRow = l * p;
                final long lastRow = (l == (nthreads - 1)) ? rowsl : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (long r = firstRow; r < lastRow; r++) {
                            long idx1 = 2 * r;
                            for (long c = 0; c < n2d2; c++) {
                                long idx0 = 2 * c;
                                long idx2 = r * rowStride + idx0;
                                a.setDouble(idx2, temp.getFloat(c * temp_stride + idx1));
                                a.setDouble(idx2 + 1, temp.getFloat(c * temp_stride + idx1 + 1));
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int l = 0; l < nthreads; l++) {
                final long firstRow = 1 + l * p;
                final long lastRow = (l == (nthreads - 1)) ? rowsl : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (long r = firstRow; r < lastRow; r++) {
                            long idx5 = r * rowStride;
                            long idx6 = (rowsl - r + 1) * rowStride;
                            for (long c = n2d2; c < columnsl; c++) {
                                long idx1 = 2 * c;
                                long idx2 = 2 * (columnsl - c);
                                a.setDouble(idx1, a.getFloat(idx2));
                                a.setDouble(idx1 + 1, -a.getFloat(idx2 + 1));
                                long idx3 = idx5 + idx1;
                                long idx4 = idx6 - idx1;
                                a.setDouble(idx3, a.getFloat(idx4));
                                a.setDouble(idx3 + 1, -a.getFloat(idx4 + 1));
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }

        } else {
            for (long r = 0; r < rowsl; r++) {
                fftColumns.realForward(a, r * columnsl);
            }
            for (long r = 0; r < rowsl; r++) {
                temp.setDouble(r, a.getFloat(r * columnsl)); //first column is always real
            }
            fftRows.realForwardFull(temp);

            for (long c = 1; c < n2d2 - 1; c++) {
                long idx0 = 2 * c;
                for (long r = 0; r < rowsl; r++) {
                    long idx1 = 2 * r;
                    long idx2 = r * columnsl + idx0;
                    temp.setDouble(c * temp_stride + idx1, a.getFloat(idx2));
                    temp.setDouble(c * temp_stride + idx1 + 1, a.getFloat(idx2 + 1));
                }
                fftRows.complexForward(temp, c * temp_stride);
            }

            if ((columnsl % 2) == 0) {
                for (long r = 0; r < rowsl; r++) {
                    temp.setDouble((n2d2 - 1) * temp_stride + r, a.getFloat(r * columnsl + 1));
                    //imaginary part = 0;
                }
                fftRows.realForwardFull(temp, (n2d2 - 1) * temp_stride);

            } else {
                for (long r = 0; r < rowsl; r++) {
                    long idx1 = 2 * r;
                    long idx2 = r * columnsl;
                    long idx3 = n2d2 - 1;
                    temp.setDouble(idx3 * temp_stride + idx1, a.getFloat(idx2 + 2 * idx3));
                    temp.setDouble(idx3 * temp_stride + idx1 + 1, a.getFloat(idx2 + 1));
                }
                fftRows.complexForward(temp, (n2d2 - 1) * temp_stride);
            }

            for (long r = 0; r < rowsl; r++) {
                long idx1 = 2 * r;
                for (long c = 0; c < n2d2; c++) {
                    long idx0 = 2 * c;
                    long idx2 = r * rowStride + idx0;
                    a.setDouble(idx2, temp.getFloat(c * temp_stride + idx1));
                    a.setDouble(idx2 + 1, temp.getFloat(c * temp_stride + idx1 + 1));
                }
            }

            //fill symmetric
            for (long r = 1; r < rowsl; r++) {
                long idx5 = r * rowStride;
                long idx6 = (rowsl - r + 1) * rowStride;
                for (long c = n2d2; c < columnsl; c++) {
                    long idx1 = 2 * c;
                    long idx2 = 2 * (columnsl - c);
                    a.setDouble(idx1, a.getFloat(idx2));
                    a.setDouble(idx1 + 1, -a.getFloat(idx2 + 1));
                    long idx3 = idx5 + idx1;
                    long idx4 = idx6 - idx1;
                    a.setDouble(idx3, a.getFloat(idx4));
                    a.setDouble(idx3 + 1, -a.getFloat(idx4 + 1));
                }
            }
        }
    }

    private void mixedRadixRealInverseFull(final float[][] a, final boolean scale)
    {
        final int n2d2 = columns / 2 + 1;
        final float[][] temp = new float[n2d2][2 * rows];

        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && useThreads && (rows >= nthreads) && (n2d2 - 2 >= nthreads)) {
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
                            fftColumns.realInverse2(a[i], 0, scale);
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int r = 0; r < rows; r++) {
                temp[0][r] = a[r][0]; //first column is always real
            }
            fftRows.realInverseFull(temp[0], scale);

            p = (n2d2 - 2) / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstColumn = 1 + l * p;
                final int lastColumn = (l == (nthreads - 1)) ? n2d2 - 1 : firstColumn + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (int c = firstColumn; c < lastColumn; c++) {
                            int idx2 = 2 * c;
                            for (int r = 0; r < rows; r++) {
                                int idx1 = 2 * r;
                                temp[c][idx1] = a[r][idx2];
                                temp[c][idx1 + 1] = a[r][idx2 + 1];
                            }
                            fftRows.complexInverse(temp[c], scale);
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }

            if ((columns % 2) == 0) {
                for (int r = 0; r < rows; r++) {
                    temp[n2d2 - 1][r] = a[r][1];
                    //imaginary part = 0;
                }
                fftRows.realInverseFull(temp[n2d2 - 1], scale);

            } else {
                for (int r = 0; r < rows; r++) {
                    int idx1 = 2 * r;
                    int idx2 = n2d2 - 1;
                    temp[idx2][idx1] = a[r][2 * idx2];
                    temp[idx2][idx1 + 1] = a[r][1];
                }
                fftRows.complexInverse(temp[n2d2 - 1], scale);

            }

            p = rows / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstRow = l * p;
                final int lastRow = (l == (nthreads - 1)) ? rows : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (int r = firstRow; r < lastRow; r++) {
                            int idx1 = 2 * r;
                            for (int c = 0; c < n2d2; c++) {
                                int idx2 = 2 * c;
                                a[r][idx2] = temp[c][idx1];
                                a[r][idx2 + 1] = temp[c][idx1 + 1];
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int l = 0; l < nthreads; l++) {
                final int firstRow = 1 + l * p;
                final int lastRow = (l == (nthreads - 1)) ? rows : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (int r = firstRow; r < lastRow; r++) {
                            int idx3 = rows - r;
                            for (int c = n2d2; c < columns; c++) {
                                int idx1 = 2 * c;
                                int idx2 = 2 * (columns - c);
                                a[0][idx1] = a[0][idx2];
                                a[0][idx1 + 1] = -a[0][idx2 + 1];
                                a[r][idx1] = a[idx3][idx2];
                                a[r][idx1 + 1] = -a[idx3][idx2 + 1];
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }

        } else {
            for (int r = 0; r < rows; r++) {
                fftColumns.realInverse2(a[r], 0, scale);
            }

            for (int r = 0; r < rows; r++) {
                temp[0][r] = a[r][0]; //first column is always real
            }
            fftRows.realInverseFull(temp[0], scale);

            for (int c = 1; c < n2d2 - 1; c++) {
                int idx2 = 2 * c;
                for (int r = 0; r < rows; r++) {
                    int idx1 = 2 * r;
                    temp[c][idx1] = a[r][idx2];
                    temp[c][idx1 + 1] = a[r][idx2 + 1];
                }
                fftRows.complexInverse(temp[c], scale);
            }

            if ((columns % 2) == 0) {
                for (int r = 0; r < rows; r++) {
                    temp[n2d2 - 1][r] = a[r][1];
                    //imaginary part = 0;
                }
                fftRows.realInverseFull(temp[n2d2 - 1], scale);

            } else {
                for (int r = 0; r < rows; r++) {
                    int idx1 = 2 * r;
                    int idx2 = n2d2 - 1;
                    temp[idx2][idx1] = a[r][2 * idx2];
                    temp[idx2][idx1 + 1] = a[r][1];
                }
                fftRows.complexInverse(temp[n2d2 - 1], scale);

            }

            for (int r = 0; r < rows; r++) {
                int idx1 = 2 * r;
                for (int c = 0; c < n2d2; c++) {
                    int idx2 = 2 * c;
                    a[r][idx2] = temp[c][idx1];
                    a[r][idx2 + 1] = temp[c][idx1 + 1];
                }
            }

            //fill symmetric
            for (int r = 1; r < rows; r++) {
                int idx3 = rows - r;
                for (int c = n2d2; c < columns; c++) {
                    int idx1 = 2 * c;
                    int idx2 = 2 * (columns - c);
                    a[0][idx1] = a[0][idx2];
                    a[0][idx1 + 1] = -a[0][idx2 + 1];
                    a[r][idx1] = a[idx3][idx2];
                    a[r][idx1 + 1] = -a[idx3][idx2 + 1];
                }
            }
        }
    }

    private void mixedRadixRealInverseFull(final float[] a, final boolean scale)
    {
        final int rowStride = 2 * columns;
        final int n2d2 = columns / 2 + 1;
        final float[][] temp = new float[n2d2][2 * rows];

        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && useThreads && (rows >= nthreads) && (n2d2 - 2 >= nthreads)) {
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
                            fftColumns.realInverse2(a, i * columns, scale);
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int r = 0; r < rows; r++) {
                temp[0][r] = a[r * columns]; //first column is always real
            }
            fftRows.realInverseFull(temp[0], scale);

            p = (n2d2 - 2) / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstColumn = 1 + l * p;
                final int lastColumn = (l == (nthreads - 1)) ? n2d2 - 1 : firstColumn + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (int c = firstColumn; c < lastColumn; c++) {
                            int idx0 = 2 * c;
                            for (int r = 0; r < rows; r++) {
                                int idx1 = 2 * r;
                                int idx2 = r * columns + idx0;
                                temp[c][idx1] = a[idx2];
                                temp[c][idx1 + 1] = a[idx2 + 1];
                            }
                            fftRows.complexInverse(temp[c], scale);
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }

            if ((columns % 2) == 0) {
                for (int r = 0; r < rows; r++) {
                    temp[n2d2 - 1][r] = a[r * columns + 1];
                    //imaginary part = 0;
                }
                fftRows.realInverseFull(temp[n2d2 - 1], scale);

            } else {
                for (int r = 0; r < rows; r++) {
                    int idx1 = 2 * r;
                    int idx2 = r * columns;
                    int idx3 = n2d2 - 1;
                    temp[idx3][idx1] = a[idx2 + 2 * idx3];
                    temp[idx3][idx1 + 1] = a[idx2 + 1];
                }
                fftRows.complexInverse(temp[n2d2 - 1], scale);
            }

            p = rows / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final int firstRow = l * p;
                final int lastRow = (l == (nthreads - 1)) ? rows : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (int r = firstRow; r < lastRow; r++) {
                            int idx1 = 2 * r;
                            for (int c = 0; c < n2d2; c++) {
                                int idx0 = 2 * c;
                                int idx2 = r * rowStride + idx0;
                                a[idx2] = temp[c][idx1];
                                a[idx2 + 1] = temp[c][idx1 + 1];
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int l = 0; l < nthreads; l++) {
                final int firstRow = 1 + l * p;
                final int lastRow = (l == (nthreads - 1)) ? rows : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (int r = firstRow; r < lastRow; r++) {
                            int idx5 = r * rowStride;
                            int idx6 = (rows - r + 1) * rowStride;
                            for (int c = n2d2; c < columns; c++) {
                                int idx1 = 2 * c;
                                int idx2 = 2 * (columns - c);
                                a[idx1] = a[idx2];
                                a[idx1 + 1] = -a[idx2 + 1];
                                int idx3 = idx5 + idx1;
                                int idx4 = idx6 - idx1;
                                a[idx3] = a[idx4];
                                a[idx3 + 1] = -a[idx4 + 1];
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {
            for (int r = 0; r < rows; r++) {
                fftColumns.realInverse2(a, r * columns, scale);
            }
            for (int r = 0; r < rows; r++) {
                temp[0][r] = a[r * columns]; //first column is always real
            }
            fftRows.realInverseFull(temp[0], scale);

            for (int c = 1; c < n2d2 - 1; c++) {
                int idx0 = 2 * c;
                for (int r = 0; r < rows; r++) {
                    int idx1 = 2 * r;
                    int idx2 = r * columns + idx0;
                    temp[c][idx1] = a[idx2];
                    temp[c][idx1 + 1] = a[idx2 + 1];
                }
                fftRows.complexInverse(temp[c], scale);
            }

            if ((columns % 2) == 0) {
                for (int r = 0; r < rows; r++) {
                    temp[n2d2 - 1][r] = a[r * columns + 1];
                    //imaginary part = 0;
                }
                fftRows.realInverseFull(temp[n2d2 - 1], scale);

            } else {
                for (int r = 0; r < rows; r++) {
                    int idx1 = 2 * r;
                    int idx2 = r * columns;
                    int idx3 = n2d2 - 1;
                    temp[idx3][idx1] = a[idx2 + 2 * idx3];
                    temp[idx3][idx1 + 1] = a[idx2 + 1];
                }
                fftRows.complexInverse(temp[n2d2 - 1], scale);
            }

            for (int r = 0; r < rows; r++) {
                int idx1 = 2 * r;
                for (int c = 0; c < n2d2; c++) {
                    int idx0 = 2 * c;
                    int idx2 = r * rowStride + idx0;
                    a[idx2] = temp[c][idx1];
                    a[idx2 + 1] = temp[c][idx1 + 1];
                }
            }

            //fill symmetric
            for (int r = 1; r < rows; r++) {
                int idx5 = r * rowStride;
                int idx6 = (rows - r + 1) * rowStride;
                for (int c = n2d2; c < columns; c++) {
                    int idx1 = 2 * c;
                    int idx2 = 2 * (columns - c);
                    a[idx1] = a[idx2];
                    a[idx1 + 1] = -a[idx2 + 1];
                    int idx3 = idx5 + idx1;
                    int idx4 = idx6 - idx1;
                    a[idx3] = a[idx4];
                    a[idx3 + 1] = -a[idx4 + 1];
                }
            }
        }
    }

    private void mixedRadixRealInverseFull(final FloatLargeArray a, final boolean scale)
    {
        final long rowStride = 2 * columnsl;
        final long n2d2 = columnsl / 2 + 1;
        final FloatLargeArray temp = new FloatLargeArray(n2d2 * 2 * rowsl);
        final long temp_stride = 2 * rowsl;

        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && useThreads && (rowsl >= nthreads) && (n2d2 - 2 >= nthreads)) {
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
                            fftColumns.realInverse2(a, i * columnsl, scale);
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (long r = 0; r < rowsl; r++) {
                temp.setDouble(r, a.getFloat(r * columnsl)); //first column is always real
            }
            fftRows.realInverseFull(temp, scale);

            p = (n2d2 - 2) / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final long firstColumn = 1 + l * p;
                final long lastColumn = (l == (nthreads - 1)) ? n2d2 - 1 : firstColumn + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (long c = firstColumn; c < lastColumn; c++) {
                            long idx0 = 2 * c;
                            for (long r = 0; r < rowsl; r++) {
                                long idx1 = 2 * r;
                                long idx2 = r * columnsl + idx0;
                                temp.setDouble(c * temp_stride + idx1, a.getFloat(idx2));
                                temp.setDouble(c * temp_stride + idx1 + 1, a.getFloat(idx2 + 1));
                            }
                            fftRows.complexInverse(temp, c * temp_stride, scale);
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }

            if ((columnsl % 2) == 0) {
                for (long r = 0; r < rowsl; r++) {
                    temp.setDouble((n2d2 - 1) * temp_stride + r, a.getFloat(r * columnsl + 1));
                    //imaginary part = 0;
                }
                fftRows.realInverseFull(temp, (n2d2 - 1) * temp_stride, scale);

            } else {
                for (long r = 0; r < rowsl; r++) {
                    long idx1 = 2 * r;
                    long idx2 = r * columnsl;
                    long idx3 = n2d2 - 1;
                    temp.setDouble(idx3 * temp_stride + idx1, a.getFloat(idx2 + 2 * idx3));
                    temp.setDouble(idx3 * temp_stride + idx1 + 1, a.getFloat(idx2 + 1));
                }
                fftRows.complexInverse(temp, (n2d2 - 1) * temp_stride, scale);
            }

            p = rowsl / nthreads;
            for (int l = 0; l < nthreads; l++) {
                final long firstRow = l * p;
                final long lastRow = (l == (nthreads - 1)) ? rowsl : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (long r = firstRow; r < lastRow; r++) {
                            long idx1 = 2 * r;
                            for (long c = 0; c < n2d2; c++) {
                                long idx0 = 2 * c;
                                long idx2 = r * rowStride + idx0;
                                a.setDouble(idx2, temp.getFloat(c * temp_stride + idx1));
                                a.setDouble(idx2 + 1, temp.getFloat(c * temp_stride + idx1 + 1));
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }

            for (int l = 0; l < nthreads; l++) {
                final long firstRow = 1 + l * p;
                final long lastRow = (l == (nthreads - 1)) ? rowsl : firstRow + p;
                futures[l] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        for (long r = firstRow; r < lastRow; r++) {
                            long idx5 = r * rowStride;
                            long idx6 = (rowsl - r + 1) * rowStride;
                            for (long c = n2d2; c < columnsl; c++) {
                                long idx1 = 2 * c;
                                long idx2 = 2 * (columnsl - c);
                                a.setDouble(idx1, a.getFloat(idx2));
                                a.setDouble(idx1 + 1, -a.getFloat(idx2 + 1));
                                long idx3 = idx5 + idx1;
                                long idx4 = idx6 - idx1;
                                a.setDouble(idx3, a.getFloat(idx4));
                                a.setDouble(idx3 + 1, -a.getFloat(idx4 + 1));
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {
            for (long r = 0; r < rowsl; r++) {
                fftColumns.realInverse2(a, r * columnsl, scale);
            }
            for (long r = 0; r < rowsl; r++) {
                temp.setDouble(r, a.getFloat(r * columnsl)); //first column is always real
            }
            fftRows.realInverseFull(temp, scale);

            for (long c = 1; c < n2d2 - 1; c++) {
                long idx0 = 2 * c;
                for (long r = 0; r < rowsl; r++) {
                    long idx1 = 2 * r;
                    long idx2 = r * columnsl + idx0;
                    temp.setDouble(c * temp_stride + idx1, a.getFloat(idx2));
                    temp.setDouble(c * temp_stride + idx1 + 1, a.getFloat(idx2 + 1));
                }
                fftRows.complexInverse(temp, c * temp_stride, scale);
            }

            if ((columnsl % 2) == 0) {
                for (long r = 0; r < rowsl; r++) {
                    temp.setDouble((n2d2 - 1) * temp_stride + r, a.getFloat(r * columnsl + 1));
                    //imaginary part = 0;
                }
                fftRows.realInverseFull(temp, (n2d2 - 1) * temp_stride, scale);

            } else {
                for (long r = 0; r < rowsl; r++) {
                    long idx1 = 2 * r;
                    long idx2 = r * columnsl;
                    long idx3 = n2d2 - 1;
                    temp.setDouble(idx3 * temp_stride + idx1, a.getFloat(idx2 + 2 * idx3));
                    temp.setDouble(idx3 * temp_stride + idx1 + 1, a.getFloat(idx2 + 1));
                }
                fftRows.complexInverse(temp, (n2d2 - 1) * temp_stride, scale);
            }

            for (long r = 0; r < rowsl; r++) {
                long idx1 = 2 * r;
                for (long c = 0; c < n2d2; c++) {
                    long idx0 = 2 * c;
                    long idx2 = r * rowStride + idx0;
                    a.setDouble(idx2, temp.getFloat(c * temp_stride + idx1));
                    a.setDouble(idx2 + 1, temp.getFloat(c * temp_stride + idx1 + 1));
                }
            }

            //fill symmetric
            for (long r = 1; r < rowsl; r++) {
                long idx5 = r * rowStride;
                long idx6 = (rowsl - r + 1) * rowStride;
                for (long c = n2d2; c < columnsl; c++) {
                    long idx1 = 2 * c;
                    long idx2 = 2 * (columnsl - c);
                    a.setDouble(idx1, a.getFloat(idx2));
                    a.setDouble(idx1 + 1, -a.getFloat(idx2 + 1));
                    long idx3 = idx5 + idx1;
                    long idx4 = idx6 - idx1;
                    a.setDouble(idx3, a.getFloat(idx4));
                    a.setDouble(idx3 + 1, -a.getFloat(idx4 + 1));
                }
            }
        }
    }

    private void rdft2d_sub(int isgn, float[] a)
    {
        int n1h, j;
        float xi;
        int idx1, idx2;

        n1h = rows >> 1;
        if (isgn < 0) {
            for (int i = 1; i < n1h; i++) {
                j = rows - i;
                idx1 = i * columns;
                idx2 = j * columns;
                xi = a[idx1] - a[idx2];
                a[idx1] += a[idx2];
                a[idx2] = xi;
                xi = a[idx2 + 1] - a[idx1 + 1];
                a[idx1 + 1] += a[idx2 + 1];
                a[idx2 + 1] = xi;
            }
        } else {
            for (int i = 1; i < n1h; i++) {
                j = rows - i;
                idx1 = i * columns;
                idx2 = j * columns;
                a[idx2] = 0.5f * (a[idx1] - a[idx2]);
                a[idx1] -= a[idx2];
                a[idx2 + 1] = 0.5f * (a[idx1 + 1] + a[idx2 + 1]);
                a[idx1 + 1] -= a[idx2 + 1];
            }
        }
    }

    private void rdft2d_sub(int isgn, FloatLargeArray a)
    {
        long n1h, j;
        float xi;
        long idx1, idx2;

        n1h = rowsl >> 1;
        if (isgn < 0) {
            for (long i = 1; i < n1h; i++) {
                j = rowsl - i;
                idx1 = i * columnsl;
                idx2 = j * columnsl;
                xi = a.getFloat(idx1) - a.getFloat(idx2);
                a.setDouble(idx1, a.getFloat(idx1) + a.getFloat(idx2));
                a.setDouble(idx2, xi);
                xi = a.getFloat(idx2 + 1) - a.getFloat(idx1 + 1);
                a.setDouble(idx1 + 1, a.getFloat(idx1 + 1) + a.getFloat(idx2 + 1));
                a.setDouble(idx2 + 1, xi);
            }
        } else {
            for (long i = 1; i < n1h; i++) {
                j = rowsl - i;
                idx1 = i * columnsl;
                idx2 = j * columnsl;
                a.setDouble(idx2, 0.5f * (a.getFloat(idx1) - a.getFloat(idx2)));
                a.setDouble(idx1, a.getFloat(idx1) - a.getFloat(idx2));
                a.setDouble(idx2 + 1, 0.5f * (a.getFloat(idx1 + 1) + a.getFloat(idx2 + 1)));
                a.setDouble(idx1 + 1, a.getFloat(idx1 + 1) - a.getFloat(idx2 + 1));
            }
        }
    }

    private void rdft2d_sub(int isgn, float[][] a)
    {
        int n1h, j;
        float xi;

        n1h = rows >> 1;
        if (isgn < 0) {
            for (int i = 1; i < n1h; i++) {
                j = rows - i;
                xi = a[i][0] - a[j][0];
                a[i][0] += a[j][0];
                a[j][0] = xi;
                xi = a[j][1] - a[i][1];
                a[i][1] += a[j][1];
                a[j][1] = xi;
            }
        } else {
            for (int i = 1; i < n1h; i++) {
                j = rows - i;
                a[j][0] = 0.5f * (a[i][0] - a[j][0]);
                a[i][0] -= a[j][0];
                a[j][1] = 0.5f * (a[i][1] + a[j][1]);
                a[i][1] -= a[j][1];
            }
        }
    }

    private void cdft2d_sub(int isgn, float[] a, boolean scale)
    {
        int idx1, idx2, idx3, idx4, idx5;
        int nt = 8 * rows;
        if (columns == 4) {
            nt >>= 1;
        } else if (columns < 4) {
            nt >>= 2;
        }
        float[] t = new float[nt];
        if (isgn == -1) {
            if (columns > 4) {
                for (int c = 0; c < columns; c += 8) {
                    for (int r = 0; r < rows; r++) {
                        idx1 = r * columns + c;
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
                        idx1 = r * columns + c;
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
                    idx1 = r * columns;
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
                    idx1 = r * columns;
                    idx2 = 2 * r;
                    idx3 = 2 * rows + 2 * r;
                    a[idx1] = t[idx2];
                    a[idx1 + 1] = t[idx2 + 1];
                    a[idx1 + 2] = t[idx3];
                    a[idx1 + 3] = t[idx3 + 1];
                }
            } else if (columns == 2) {
                for (int r = 0; r < rows; r++) {
                    idx1 = r * columns;
                    idx2 = 2 * r;
                    t[idx2] = a[idx1];
                    t[idx2 + 1] = a[idx1 + 1];
                }
                fftRows.complexForward(t, 0);
                for (int r = 0; r < rows; r++) {
                    idx1 = r * columns;
                    idx2 = 2 * r;
                    a[idx1] = t[idx2];
                    a[idx1 + 1] = t[idx2 + 1];
                }
            }
        } else if (columns > 4) {
            for (int c = 0; c < columns; c += 8) {
                for (int r = 0; r < rows; r++) {
                    idx1 = r * columns + c;
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
                    idx1 = r * columns + c;
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
                idx1 = r * columns;
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
                idx1 = r * columns;
                idx2 = 2 * r;
                idx3 = 2 * rows + 2 * r;
                a[idx1] = t[idx2];
                a[idx1 + 1] = t[idx2 + 1];
                a[idx1 + 2] = t[idx3];
                a[idx1 + 3] = t[idx3 + 1];
            }
        } else if (columns == 2) {
            for (int r = 0; r < rows; r++) {
                idx1 = r * columns;
                idx2 = 2 * r;
                t[idx2] = a[idx1];
                t[idx2 + 1] = a[idx1 + 1];
            }
            fftRows.complexInverse(t, 0, scale);
            for (int r = 0; r < rows; r++) {
                idx1 = r * columns;
                idx2 = 2 * r;
                a[idx1] = t[idx2];
                a[idx1 + 1] = t[idx2 + 1];
            }
        }
    }

    private void cdft2d_sub(int isgn, FloatLargeArray a, boolean scale)
    {
        long idx1, idx2, idx3, idx4, idx5;
        long nt = 8 * rowsl;
        if (columnsl == 4) {
            nt >>= 1;
        } else if (columnsl < 4) {
            nt >>= 2;
        }
        FloatLargeArray t = new FloatLargeArray(nt);
        if (isgn == -1) {
            if (columnsl > 4) {
                for (long c = 0; c < columnsl; c += 8) {
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = r * columnsl + c;
                        idx2 = 2 * r;
                        idx3 = 2 * rowsl + 2 * r;
                        idx4 = idx3 + 2 * rowsl;
                        idx5 = idx4 + 2 * rowsl;
                        t.setDouble(idx2, a.getFloat(idx1));
                        t.setDouble(idx2 + 1, a.getFloat(idx1 + 1));
                        t.setDouble(idx3, a.getFloat(idx1 + 2));
                        t.setDouble(idx3 + 1, a.getFloat(idx1 + 3));
                        t.setDouble(idx4, a.getFloat(idx1 + 4));
                        t.setDouble(idx4 + 1, a.getFloat(idx1 + 5));
                        t.setDouble(idx5, a.getFloat(idx1 + 6));
                        t.setDouble(idx5 + 1, a.getFloat(idx1 + 7));
                    }
                    fftRows.complexForward(t, 0);
                    fftRows.complexForward(t, 2 * rowsl);
                    fftRows.complexForward(t, 4 * rowsl);
                    fftRows.complexForward(t, 6 * rowsl);
                    for (long r = 0; r < rowsl; r++) {
                        idx1 = r * columnsl + c;
                        idx2 = 2 * r;
                        idx3 = 2 * rowsl + 2 * r;
                        idx4 = idx3 + 2 * rowsl;
                        idx5 = idx4 + 2 * rowsl;
                        a.setDouble(idx1, t.getFloat(idx2));
                        a.setDouble(idx1 + 1, t.getFloat(idx2 + 1));
                        a.setDouble(idx1 + 2, t.getFloat(idx3));
                        a.setDouble(idx1 + 3, t.getFloat(idx3 + 1));
                        a.setDouble(idx1 + 4, t.getFloat(idx4));
                        a.setDouble(idx1 + 5, t.getFloat(idx4 + 1));
                        a.setDouble(idx1 + 6, t.getFloat(idx5));
                        a.setDouble(idx1 + 7, t.getFloat(idx5 + 1));
                    }
                }
            } else if (columnsl == 4) {
                for (long r = 0; r < rowsl; r++) {
                    idx1 = r * columnsl;
                    idx2 = 2 * r;
                    idx3 = 2 * rowsl + 2 * r;
                    t.setDouble(idx2, a.getFloat(idx1));
                    t.setDouble(idx2 + 1, a.getFloat(idx1 + 1));
                    t.setDouble(idx3, a.getFloat(idx1 + 2));
                    t.setDouble(idx3 + 1, a.getFloat(idx1 + 3));
                }
                fftRows.complexForward(t, 0);
                fftRows.complexForward(t, 2 * rowsl);
                for (long r = 0; r < rowsl; r++) {
                    idx1 = r * columnsl;
                    idx2 = 2 * r;
                    idx3 = 2 * rowsl + 2 * r;
                    a.setDouble(idx1, t.getFloat(idx2));
                    a.setDouble(idx1 + 1, t.getFloat(idx2 + 1));
                    a.setDouble(idx1 + 2, t.getFloat(idx3));
                    a.setDouble(idx1 + 3, t.getFloat(idx3 + 1));
                }
            } else if (columnsl == 2) {
                for (long r = 0; r < rowsl; r++) {
                    idx1 = r * columnsl;
                    idx2 = 2 * r;
                    t.setDouble(idx2, a.getFloat(idx1));
                    t.setDouble(idx2 + 1, a.getFloat(idx1 + 1));
                }
                fftRows.complexForward(t, 0);
                for (long r = 0; r < rowsl; r++) {
                    idx1 = r * columnsl;
                    idx2 = 2 * r;
                    a.setDouble(idx1, t.getFloat(idx2));
                    a.setDouble(idx1 + 1, t.getFloat(idx2 + 1));
                }
            }
        } else if (columnsl > 4) {
            for (long c = 0; c < columnsl; c += 8) {
                for (long r = 0; r < rowsl; r++) {
                    idx1 = r * columnsl + c;
                    idx2 = 2 * r;
                    idx3 = 2 * rowsl + 2 * r;
                    idx4 = idx3 + 2 * rowsl;
                    idx5 = idx4 + 2 * rowsl;
                    t.setDouble(idx2, a.getFloat(idx1));
                    t.setDouble(idx2 + 1, a.getFloat(idx1 + 1));
                    t.setDouble(idx3, a.getFloat(idx1 + 2));
                    t.setDouble(idx3 + 1, a.getFloat(idx1 + 3));
                    t.setDouble(idx4, a.getFloat(idx1 + 4));
                    t.setDouble(idx4 + 1, a.getFloat(idx1 + 5));
                    t.setDouble(idx5, a.getFloat(idx1 + 6));
                    t.setDouble(idx5 + 1, a.getFloat(idx1 + 7));
                }
                fftRows.complexInverse(t, 0, scale);
                fftRows.complexInverse(t, 2 * rowsl, scale);
                fftRows.complexInverse(t, 4 * rowsl, scale);
                fftRows.complexInverse(t, 6 * rowsl, scale);
                for (long r = 0; r < rowsl; r++) {
                    idx1 = r * columnsl + c;
                    idx2 = 2 * r;
                    idx3 = 2 * rowsl + 2 * r;
                    idx4 = idx3 + 2 * rowsl;
                    idx5 = idx4 + 2 * rowsl;
                    a.setDouble(idx1, t.getFloat(idx2));
                    a.setDouble(idx1 + 1, t.getFloat(idx2 + 1));
                    a.setDouble(idx1 + 2, t.getFloat(idx3));
                    a.setDouble(idx1 + 3, t.getFloat(idx3 + 1));
                    a.setDouble(idx1 + 4, t.getFloat(idx4));
                    a.setDouble(idx1 + 5, t.getFloat(idx4 + 1));
                    a.setDouble(idx1 + 6, t.getFloat(idx5));
                    a.setDouble(idx1 + 7, t.getFloat(idx5 + 1));
                }
            }
        } else if (columnsl == 4) {
            for (long r = 0; r < rowsl; r++) {
                idx1 = r * columnsl;
                idx2 = 2 * r;
                idx3 = 2 * rowsl + 2 * r;
                t.setDouble(idx2, a.getFloat(idx1));
                t.setDouble(idx2 + 1, a.getFloat(idx1 + 1));
                t.setDouble(idx3, a.getFloat(idx1 + 2));
                t.setDouble(idx3 + 1, a.getFloat(idx1 + 3));
            }
            fftRows.complexInverse(t, 0, scale);
            fftRows.complexInverse(t, 2 * rowsl, scale);
            for (long r = 0; r < rowsl; r++) {
                idx1 = r * columnsl;
                idx2 = 2 * r;
                idx3 = 2 * rowsl + 2 * r;
                a.setDouble(idx1, t.getFloat(idx2));
                a.setDouble(idx1 + 1, t.getFloat(idx2 + 1));
                a.setDouble(idx1 + 2, t.getFloat(idx3));
                a.setDouble(idx1 + 3, t.getFloat(idx3 + 1));
            }
        } else if (columnsl == 2) {
            for (long r = 0; r < rowsl; r++) {
                idx1 = r * columnsl;
                idx2 = 2 * r;
                t.setDouble(idx2, a.getFloat(idx1));
                t.setDouble(idx2 + 1, a.getFloat(idx1 + 1));
            }
            fftRows.complexInverse(t, 0, scale);
            for (long r = 0; r < rowsl; r++) {
                idx1 = r * columnsl;
                idx2 = 2 * r;
                a.setDouble(idx1, t.getFloat(idx2));
                a.setDouble(idx1 + 1, t.getFloat(idx2 + 1));
            }
        }
    }

    private void cdft2d_sub(int isgn, float[][] a, boolean scale)
    {
        int idx2, idx3, idx4, idx5;
        int nt = 8 * rows;
        if (columns == 4) {
            nt >>= 1;
        } else if (columns < 4) {
            nt >>= 2;
        }
        float[] t = new float[nt];
        if (isgn == -1) {
            if (columns > 4) {
                for (int c = 0; c < columns; c += 8) {
                    for (int r = 0; r < rows; r++) {
                        idx2 = 2 * r;
                        idx3 = 2 * rows + 2 * r;
                        idx4 = idx3 + 2 * rows;
                        idx5 = idx4 + 2 * rows;
                        t[idx2] = a[r][c];
                        t[idx2 + 1] = a[r][c + 1];
                        t[idx3] = a[r][c + 2];
                        t[idx3 + 1] = a[r][c + 3];
                        t[idx4] = a[r][c + 4];
                        t[idx4 + 1] = a[r][c + 5];
                        t[idx5] = a[r][c + 6];
                        t[idx5 + 1] = a[r][c + 7];
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
                        a[r][c] = t[idx2];
                        a[r][c + 1] = t[idx2 + 1];
                        a[r][c + 2] = t[idx3];
                        a[r][c + 3] = t[idx3 + 1];
                        a[r][c + 4] = t[idx4];
                        a[r][c + 5] = t[idx4 + 1];
                        a[r][c + 6] = t[idx5];
                        a[r][c + 7] = t[idx5 + 1];
                    }
                }
            } else if (columns == 4) {
                for (int r = 0; r < rows; r++) {
                    idx2 = 2 * r;
                    idx3 = 2 * rows + 2 * r;
                    t[idx2] = a[r][0];
                    t[idx2 + 1] = a[r][1];
                    t[idx3] = a[r][2];
                    t[idx3 + 1] = a[r][3];
                }
                fftRows.complexForward(t, 0);
                fftRows.complexForward(t, 2 * rows);
                for (int r = 0; r < rows; r++) {
                    idx2 = 2 * r;
                    idx3 = 2 * rows + 2 * r;
                    a[r][0] = t[idx2];
                    a[r][1] = t[idx2 + 1];
                    a[r][2] = t[idx3];
                    a[r][3] = t[idx3 + 1];
                }
            } else if (columns == 2) {
                for (int r = 0; r < rows; r++) {
                    idx2 = 2 * r;
                    t[idx2] = a[r][0];
                    t[idx2 + 1] = a[r][1];
                }
                fftRows.complexForward(t, 0);
                for (int r = 0; r < rows; r++) {
                    idx2 = 2 * r;
                    a[r][0] = t[idx2];
                    a[r][1] = t[idx2 + 1];
                }
            }
        } else if (columns > 4) {
            for (int c = 0; c < columns; c += 8) {
                for (int r = 0; r < rows; r++) {
                    idx2 = 2 * r;
                    idx3 = 2 * rows + 2 * r;
                    idx4 = idx3 + 2 * rows;
                    idx5 = idx4 + 2 * rows;
                    t[idx2] = a[r][c];
                    t[idx2 + 1] = a[r][c + 1];
                    t[idx3] = a[r][c + 2];
                    t[idx3 + 1] = a[r][c + 3];
                    t[idx4] = a[r][c + 4];
                    t[idx4 + 1] = a[r][c + 5];
                    t[idx5] = a[r][c + 6];
                    t[idx5 + 1] = a[r][c + 7];
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
                    a[r][c] = t[idx2];
                    a[r][c + 1] = t[idx2 + 1];
                    a[r][c + 2] = t[idx3];
                    a[r][c + 3] = t[idx3 + 1];
                    a[r][c + 4] = t[idx4];
                    a[r][c + 5] = t[idx4 + 1];
                    a[r][c + 6] = t[idx5];
                    a[r][c + 7] = t[idx5 + 1];
                }
            }
        } else if (columns == 4) {
            for (int r = 0; r < rows; r++) {
                idx2 = 2 * r;
                idx3 = 2 * rows + 2 * r;
                t[idx2] = a[r][0];
                t[idx2 + 1] = a[r][1];
                t[idx3] = a[r][2];
                t[idx3 + 1] = a[r][3];
            }
            fftRows.complexInverse(t, 0, scale);
            fftRows.complexInverse(t, 2 * rows, scale);
            for (int r = 0; r < rows; r++) {
                idx2 = 2 * r;
                idx3 = 2 * rows + 2 * r;
                a[r][0] = t[idx2];
                a[r][1] = t[idx2 + 1];
                a[r][2] = t[idx3];
                a[r][3] = t[idx3 + 1];
            }
        } else if (columns == 2) {
            for (int r = 0; r < rows; r++) {
                idx2 = 2 * r;
                t[idx2] = a[r][0];
                t[idx2 + 1] = a[r][1];
            }
            fftRows.complexInverse(t, 0, scale);
            for (int r = 0; r < rows; r++) {
                idx2 = 2 * r;
                a[r][0] = t[idx2];
                a[r][1] = t[idx2 + 1];
            }
        }
    }

    private void xdft2d0_subth1(final int icr, final int isgn, final float[] a, final boolean scale)
    {
        final int nthreads = ConcurrencyUtils.getNumberOfThreads() > rows ? rows : ConcurrencyUtils.getNumberOfThreads();

        Future<?>[] futures = new Future[nthreads];
        for (int i = 0; i < nthreads; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {
                public void run()
                {
                    if (icr == 0) {
                        if (isgn == -1) {
                            for (int r = n0; r < rows; r += nthreads) {
                                fftColumns.complexForward(a, r * columns);
                            }
                        } else {
                            for (int r = n0; r < rows; r += nthreads) {
                                fftColumns.complexInverse(a, r * columns, scale);
                            }
                        }
                    } else if (isgn == 1) {
                        for (int r = n0; r < rows; r += nthreads) {
                            fftColumns.realForward(a, r * columns);
                        }
                    } else {
                        for (int r = n0; r < rows; r += nthreads) {
                            fftColumns.realInverse(a, r * columns, scale);
                        }
                    }
                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void xdft2d0_subth1(final long icr, final int isgn, final FloatLargeArray a, final boolean scale)
    {
        final int nthreads = (int) (ConcurrencyUtils.getNumberOfThreads() > rowsl ? rowsl : ConcurrencyUtils.getNumberOfThreads());

        Future<?>[] futures = new Future[nthreads];
        for (int i = 0; i < nthreads; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {
                public void run()
                {
                    if (icr == 0) {
                        if (isgn == -1) {
                            for (long r = n0; r < rowsl; r += nthreads) {
                                fftColumns.complexForward(a, r * columnsl);
                            }
                        } else {
                            for (long r = n0; r < rowsl; r += nthreads) {
                                fftColumns.complexInverse(a, r * columnsl, scale);
                            }
                        }
                    } else if (isgn == 1) {
                        for (long r = n0; r < rowsl; r += nthreads) {
                            fftColumns.realForward(a, r * columnsl);
                        }
                    } else {
                        for (long r = n0; r < rowsl; r += nthreads) {
                            fftColumns.realInverse(a, r * columnsl, scale);
                        }
                    }
                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void xdft2d0_subth2(final int icr, final int isgn, final float[] a, final boolean scale)
    {
        final int nthreads = ConcurrencyUtils.getNumberOfThreads() > rows ? rows : ConcurrencyUtils.getNumberOfThreads();

        Future<?>[] futures = new Future[nthreads];
        for (int i = 0; i < nthreads; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {
                public void run()
                {
                    if (icr == 0) {
                        if (isgn == -1) {
                            for (int r = n0; r < rows; r += nthreads) {
                                fftColumns.complexForward(a, r * columns);
                            }
                        } else {
                            for (int r = n0; r < rows; r += nthreads) {
                                fftColumns.complexInverse(a, r * columns, scale);
                            }
                        }
                    } else if (isgn == 1) {
                        for (int r = n0; r < rows; r += nthreads) {
                            fftColumns.realForward(a, r * columns);
                        }
                    } else {
                        for (int r = n0; r < rows; r += nthreads) {
                            fftColumns.realInverse2(a, r * columns, scale);
                        }
                    }
                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void xdft2d0_subth2(final long icr, final int isgn, final FloatLargeArray a, final boolean scale)
    {
        final int nthreads = ConcurrencyUtils.getNumberOfThreads() > rows ? rows : ConcurrencyUtils.getNumberOfThreads();

        Future<?>[] futures = new Future[nthreads];
        for (int i = 0; i < nthreads; i++) {
            final long n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {
                public void run()
                {
                    if (icr == 0) {
                        if (isgn == -1) {
                            for (long r = n0; r < rowsl; r += nthreads) {
                                fftColumns.complexForward(a, r * columnsl);
                            }
                        } else {
                            for (long r = n0; r < rowsl; r += nthreads) {
                                fftColumns.complexInverse(a, r * columnsl, scale);
                            }
                        }
                    } else if (isgn == 1) {
                        for (long r = n0; r < rowsl; r += nthreads) {
                            fftColumns.realForward(a, r * columnsl);
                        }
                    } else {
                        for (long r = n0; r < rowsl; r += nthreads) {
                            fftColumns.realInverse2(a, r * columnsl, scale);
                        }
                    }
                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void xdft2d0_subth1(final int icr, final int isgn, final float[][] a, final boolean scale)
    {
        final int nthreads = ConcurrencyUtils.getNumberOfThreads() > rows ? rows : ConcurrencyUtils.getNumberOfThreads();

        Future<?>[] futures = new Future[nthreads];
        for (int i = 0; i < nthreads; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {
                public void run()
                {
                    if (icr == 0) {
                        if (isgn == -1) {
                            for (int r = n0; r < rows; r += nthreads) {
                                fftColumns.complexForward(a[r]);
                            }
                        } else {
                            for (int r = n0; r < rows; r += nthreads) {
                                fftColumns.complexInverse(a[r], scale);
                            }
                        }
                    } else if (isgn == 1) {
                        for (int r = n0; r < rows; r += nthreads) {
                            fftColumns.realForward(a[r]);
                        }
                    } else {
                        for (int r = n0; r < rows; r += nthreads) {
                            fftColumns.realInverse(a[r], scale);
                        }
                    }
                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void xdft2d0_subth2(final int icr, final int isgn, final float[][] a, final boolean scale)
    {
        final int nthreads = ConcurrencyUtils.getNumberOfThreads() > rows ? rows : ConcurrencyUtils.getNumberOfThreads();

        Future<?>[] futures = new Future[nthreads];
        for (int i = 0; i < nthreads; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {
                public void run()
                {
                    if (icr == 0) {
                        if (isgn == -1) {
                            for (int r = n0; r < rows; r += nthreads) {
                                fftColumns.complexForward(a[r]);
                            }
                        } else {
                            for (int r = n0; r < rows; r += nthreads) {
                                fftColumns.complexInverse(a[r], scale);
                            }
                        }
                    } else if (isgn == 1) {
                        for (int r = n0; r < rows; r += nthreads) {
                            fftColumns.realForward(a[r]);
                        }
                    } else {
                        for (int r = n0; r < rows; r += nthreads) {
                            fftColumns.realInverse2(a[r], 0, scale);
                        }
                    }
                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void cdft2d_subth(final int isgn, final float[] a, final boolean scale)
    {
        int nthread = min(columns / 2, ConcurrencyUtils.getNumberOfThreads());
        int nt = 8 * rows;
        if (columns == 4) {
            nt >>= 1;
        } else if (columns < 4) {
            nt >>= 2;
        }
        final int ntf = nt;
        Future<?>[] futures = new Future[nthread];
        final int nthreads = nthread;
        for (int i = 0; i < nthread; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {
                public void run()
                {
                    int idx1, idx2, idx3, idx4, idx5;
                    float[] t = new float[ntf];
                    if (isgn == -1) {
                        if (columns > 4 * nthreads) {
                            for (int c = 8 * n0; c < columns; c += 8 * nthreads) {
                                for (int r = 0; r < rows; r++) {
                                    idx1 = r * columns + c;
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
                                    idx1 = r * columns + c;
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
                        } else if (columns == 4 * nthreads) {
                            for (int r = 0; r < rows; r++) {
                                idx1 = r * columns + 4 * n0;
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
                                idx1 = r * columns + 4 * n0;
                                idx2 = 2 * r;
                                idx3 = 2 * rows + 2 * r;
                                a[idx1] = t[idx2];
                                a[idx1 + 1] = t[idx2 + 1];
                                a[idx1 + 2] = t[idx3];
                                a[idx1 + 3] = t[idx3 + 1];
                            }
                        } else if (columns == 2 * nthreads) {
                            for (int r = 0; r < rows; r++) {
                                idx1 = r * columns + 2 * n0;
                                idx2 = 2 * r;
                                t[idx2] = a[idx1];
                                t[idx2 + 1] = a[idx1 + 1];
                            }
                            fftRows.complexForward(t, 0);
                            for (int r = 0; r < rows; r++) {
                                idx1 = r * columns + 2 * n0;
                                idx2 = 2 * r;
                                a[idx1] = t[idx2];
                                a[idx1 + 1] = t[idx2 + 1];
                            }
                        }
                    } else if (columns > 4 * nthreads) {
                        for (int c = 8 * n0; c < columns; c += 8 * nthreads) {
                            for (int r = 0; r < rows; r++) {
                                idx1 = r * columns + c;
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
                                idx1 = r * columns + c;
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
                    } else if (columns == 4 * nthreads) {
                        for (int r = 0; r < rows; r++) {
                            idx1 = r * columns + 4 * n0;
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
                            idx1 = r * columns + 4 * n0;
                            idx2 = 2 * r;
                            idx3 = 2 * rows + 2 * r;
                            a[idx1] = t[idx2];
                            a[idx1 + 1] = t[idx2 + 1];
                            a[idx1 + 2] = t[idx3];
                            a[idx1 + 3] = t[idx3 + 1];
                        }
                    } else if (columns == 2 * nthreads) {
                        for (int r = 0; r < rows; r++) {
                            idx1 = r * columns + 2 * n0;
                            idx2 = 2 * r;
                            t[idx2] = a[idx1];
                            t[idx2 + 1] = a[idx1 + 1];
                        }
                        fftRows.complexInverse(t, 0, scale);
                        for (int r = 0; r < rows; r++) {
                            idx1 = r * columns + 2 * n0;
                            idx2 = 2 * r;
                            a[idx1] = t[idx2];
                            a[idx1 + 1] = t[idx2 + 1];
                        }
                    }
                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void cdft2d_subth(final int isgn, final FloatLargeArray a, final boolean scale)
    {
        int nthread = (int) min(columnsl / 2, ConcurrencyUtils.getNumberOfThreads());
        long nt = 8 * rowsl;
        if (columnsl == 4) {
            nt >>= 1;
        } else if (columnsl < 4) {
            nt >>= 2;
        }
        final long ntf = nt;
        Future<?>[] futures = new Future[nthread];
        final int nthreads = nthread;
        for (int i = 0; i < nthread; i++) {
            final long n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {
                public void run()
                {
                    long idx1, idx2, idx3, idx4, idx5;
                    FloatLargeArray t = new FloatLargeArray(ntf);
                    if (isgn == -1) {
                        if (columnsl > 4 * nthreads) {
                            for (long c = 8 * n0; c < columnsl; c += 8 * nthreads) {
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = r * columnsl + c;
                                    idx2 = 2 * r;
                                    idx3 = 2 * rowsl + 2 * r;
                                    idx4 = idx3 + 2 * rowsl;
                                    idx5 = idx4 + 2 * rowsl;
                                    t.setDouble(idx2, a.getFloat(idx1));
                                    t.setDouble(idx2 + 1, a.getFloat(idx1 + 1));
                                    t.setDouble(idx3, a.getFloat(idx1 + 2));
                                    t.setDouble(idx3 + 1, a.getFloat(idx1 + 3));
                                    t.setDouble(idx4, a.getFloat(idx1 + 4));
                                    t.setDouble(idx4 + 1, a.getFloat(idx1 + 5));
                                    t.setDouble(idx5, a.getFloat(idx1 + 6));
                                    t.setDouble(idx5 + 1, a.getFloat(idx1 + 7));
                                }
                                fftRows.complexForward(t, 0);
                                fftRows.complexForward(t, 2 * rowsl);
                                fftRows.complexForward(t, 4 * rowsl);
                                fftRows.complexForward(t, 6 * rowsl);
                                for (long r = 0; r < rowsl; r++) {
                                    idx1 = r * columnsl + c;
                                    idx2 = 2 * r;
                                    idx3 = 2 * rowsl + 2 * r;
                                    idx4 = idx3 + 2 * rowsl;
                                    idx5 = idx4 + 2 * rowsl;
                                    a.setDouble(idx1, t.getFloat(idx2));
                                    a.setDouble(idx1 + 1, t.getFloat(idx2 + 1));
                                    a.setDouble(idx1 + 2, t.getFloat(idx3));
                                    a.setDouble(idx1 + 3, t.getFloat(idx3 + 1));
                                    a.setDouble(idx1 + 4, t.getFloat(idx4));
                                    a.setDouble(idx1 + 5, t.getFloat(idx4 + 1));
                                    a.setDouble(idx1 + 6, t.getFloat(idx5));
                                    a.setDouble(idx1 + 7, t.getFloat(idx5 + 1));
                                }
                            }
                        } else if (columnsl == 4 * nthreads) {
                            for (long r = 0; r < rowsl; r++) {
                                idx1 = r * columnsl + 4 * n0;
                                idx2 = 2 * r;
                                idx3 = 2 * rowsl + 2 * r;
                                t.setDouble(idx2, a.getFloat(idx1));
                                t.setDouble(idx2 + 1, a.getFloat(idx1 + 1));
                                t.setDouble(idx3, a.getFloat(idx1 + 2));
                                t.setDouble(idx3 + 1, a.getFloat(idx1 + 3));
                            }
                            fftRows.complexForward(t, 0);
                            fftRows.complexForward(t, 2 * rowsl);
                            for (long r = 0; r < rowsl; r++) {
                                idx1 = r * columnsl + 4 * n0;
                                idx2 = 2 * r;
                                idx3 = 2 * rowsl + 2 * r;
                                a.setDouble(idx1, t.getFloat(idx2));
                                a.setDouble(idx1 + 1, t.getFloat(idx2 + 1));
                                a.setDouble(idx1 + 2, t.getFloat(idx3));
                                a.setDouble(idx1 + 3, t.getFloat(idx3 + 1));
                            }
                        } else if (columnsl == 2 * nthreads) {
                            for (long r = 0; r < rowsl; r++) {
                                idx1 = r * columnsl + 2 * n0;
                                idx2 = 2 * r;
                                t.setDouble(idx2, a.getFloat(idx1));
                                t.setDouble(idx2 + 1, a.getFloat(idx1 + 1));
                            }
                            fftRows.complexForward(t, 0);
                            for (long r = 0; r < rowsl; r++) {
                                idx1 = r * columnsl + 2 * n0;
                                idx2 = 2 * r;
                                a.setDouble(idx1, t.getFloat(idx2));
                                a.setDouble(idx1 + 1, t.getFloat(idx2 + 1));
                            }
                        }
                    } else if (columnsl > 4 * nthreads) {
                        for (long c = 8 * n0; c < columnsl; c += 8 * nthreads) {
                            for (long r = 0; r < rowsl; r++) {
                                idx1 = r * columnsl + c;
                                idx2 = 2 * r;
                                idx3 = 2 * rowsl + 2 * r;
                                idx4 = idx3 + 2 * rowsl;
                                idx5 = idx4 + 2 * rowsl;
                                t.setDouble(idx2, a.getFloat(idx1));
                                t.setDouble(idx2 + 1, a.getFloat(idx1 + 1));
                                t.setDouble(idx3, a.getFloat(idx1 + 2));
                                t.setDouble(idx3 + 1, a.getFloat(idx1 + 3));
                                t.setDouble(idx4, a.getFloat(idx1 + 4));
                                t.setDouble(idx4 + 1, a.getFloat(idx1 + 5));
                                t.setDouble(idx5, a.getFloat(idx1 + 6));
                                t.setDouble(idx5 + 1, a.getFloat(idx1 + 7));
                            }
                            fftRows.complexInverse(t, 0, scale);
                            fftRows.complexInverse(t, 2 * rowsl, scale);
                            fftRows.complexInverse(t, 4 * rowsl, scale);
                            fftRows.complexInverse(t, 6 * rowsl, scale);
                            for (long r = 0; r < rowsl; r++) {
                                idx1 = r * columnsl + c;
                                idx2 = 2 * r;
                                idx3 = 2 * rowsl + 2 * r;
                                idx4 = idx3 + 2 * rowsl;
                                idx5 = idx4 + 2 * rowsl;
                                a.setDouble(idx1, t.getFloat(idx2));
                                a.setDouble(idx1 + 1, t.getFloat(idx2 + 1));
                                a.setDouble(idx1 + 2, t.getFloat(idx3));
                                a.setDouble(idx1 + 3, t.getFloat(idx3 + 1));
                                a.setDouble(idx1 + 4, t.getFloat(idx4));
                                a.setDouble(idx1 + 5, t.getFloat(idx4 + 1));
                                a.setDouble(idx1 + 6, t.getFloat(idx5));
                                a.setDouble(idx1 + 7, t.getFloat(idx5 + 1));
                            }
                        }
                    } else if (columnsl == 4 * nthreads) {
                        for (long r = 0; r < rowsl; r++) {
                            idx1 = r * columnsl + 4 * n0;
                            idx2 = 2 * r;
                            idx3 = 2 * rowsl + 2 * r;
                            t.setDouble(idx2, a.getFloat(idx1));
                            t.setDouble(idx2 + 1, a.getFloat(idx1 + 1));
                            t.setDouble(idx3, a.getFloat(idx1 + 2));
                            t.setDouble(idx3 + 1, a.getFloat(idx1 + 3));
                        }
                        fftRows.complexInverse(t, 0, scale);
                        fftRows.complexInverse(t, 2 * rowsl, scale);
                        for (long r = 0; r < rowsl; r++) {
                            idx1 = r * columnsl + 4 * n0;
                            idx2 = 2 * r;
                            idx3 = 2 * rowsl + 2 * r;
                            a.setDouble(idx1, t.getFloat(idx2));
                            a.setDouble(idx1 + 1, t.getFloat(idx2 + 1));
                            a.setDouble(idx1 + 2, t.getFloat(idx3));
                            a.setDouble(idx1 + 3, t.getFloat(idx3 + 1));
                        }
                    } else if (columnsl == 2 * nthreads) {
                        for (long r = 0; r < rowsl; r++) {
                            idx1 = r * columnsl + 2 * n0;
                            idx2 = 2 * r;
                            t.setDouble(idx2, a.getFloat(idx1));
                            t.setDouble(idx2 + 1, a.getFloat(idx1 + 1));
                        }
                        fftRows.complexInverse(t, 0, scale);
                        for (long r = 0; r < rowsl; r++) {
                            idx1 = r * columnsl + 2 * n0;
                            idx2 = 2 * r;
                            a.setDouble(idx1, t.getFloat(idx2));
                            a.setDouble(idx1 + 1, t.getFloat(idx2 + 1));
                        }
                    }
                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void cdft2d_subth(final int isgn, final float[][] a, final boolean scale)
    {
        int nthread = min(columns / 2, ConcurrencyUtils.getNumberOfThreads());
        int nt = 8 * rows;
        if (columns == 4) {
            nt >>= 1;
        } else if (columns < 4) {
            nt >>= 2;
        }
        final int ntf = nt;
        Future<?>[] futures = new Future[nthread];
        final int nthreads = nthread;
        for (int i = 0; i < nthreads; i++) {
            final int n0 = i;
            futures[i] = ConcurrencyUtils.submit(new Runnable()
            {
                public void run()
                {
                    int idx2, idx3, idx4, idx5;
                    float[] t = new float[ntf];
                    if (isgn == -1) {
                        if (columns > 4 * nthreads) {
                            for (int c = 8 * n0; c < columns; c += 8 * nthreads) {
                                for (int r = 0; r < rows; r++) {
                                    idx2 = 2 * r;
                                    idx3 = 2 * rows + 2 * r;
                                    idx4 = idx3 + 2 * rows;
                                    idx5 = idx4 + 2 * rows;
                                    t[idx2] = a[r][c];
                                    t[idx2 + 1] = a[r][c + 1];
                                    t[idx3] = a[r][c + 2];
                                    t[idx3 + 1] = a[r][c + 3];
                                    t[idx4] = a[r][c + 4];
                                    t[idx4 + 1] = a[r][c + 5];
                                    t[idx5] = a[r][c + 6];
                                    t[idx5 + 1] = a[r][c + 7];
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
                                    a[r][c] = t[idx2];
                                    a[r][c + 1] = t[idx2 + 1];
                                    a[r][c + 2] = t[idx3];
                                    a[r][c + 3] = t[idx3 + 1];
                                    a[r][c + 4] = t[idx4];
                                    a[r][c + 5] = t[idx4 + 1];
                                    a[r][c + 6] = t[idx5];
                                    a[r][c + 7] = t[idx5 + 1];
                                }
                            }
                        } else if (columns == 4 * nthreads) {
                            for (int r = 0; r < rows; r++) {
                                idx2 = 2 * r;
                                idx3 = 2 * rows + 2 * r;
                                t[idx2] = a[r][4 * n0];
                                t[idx2 + 1] = a[r][4 * n0 + 1];
                                t[idx3] = a[r][4 * n0 + 2];
                                t[idx3 + 1] = a[r][4 * n0 + 3];
                            }
                            fftRows.complexForward(t, 0);
                            fftRows.complexForward(t, 2 * rows);
                            for (int r = 0; r < rows; r++) {
                                idx2 = 2 * r;
                                idx3 = 2 * rows + 2 * r;
                                a[r][4 * n0] = t[idx2];
                                a[r][4 * n0 + 1] = t[idx2 + 1];
                                a[r][4 * n0 + 2] = t[idx3];
                                a[r][4 * n0 + 3] = t[idx3 + 1];
                            }
                        } else if (columns == 2 * nthreads) {
                            for (int r = 0; r < rows; r++) {
                                idx2 = 2 * r;
                                t[idx2] = a[r][2 * n0];
                                t[idx2 + 1] = a[r][2 * n0 + 1];
                            }
                            fftRows.complexForward(t, 0);
                            for (int r = 0; r < rows; r++) {
                                idx2 = 2 * r;
                                a[r][2 * n0] = t[idx2];
                                a[r][2 * n0 + 1] = t[idx2 + 1];
                            }
                        }
                    } else if (columns > 4 * nthreads) {
                        for (int c = 8 * n0; c < columns; c += 8 * nthreads) {
                            for (int r = 0; r < rows; r++) {
                                idx2 = 2 * r;
                                idx3 = 2 * rows + 2 * r;
                                idx4 = idx3 + 2 * rows;
                                idx5 = idx4 + 2 * rows;
                                t[idx2] = a[r][c];
                                t[idx2 + 1] = a[r][c + 1];
                                t[idx3] = a[r][c + 2];
                                t[idx3 + 1] = a[r][c + 3];
                                t[idx4] = a[r][c + 4];
                                t[idx4 + 1] = a[r][c + 5];
                                t[idx5] = a[r][c + 6];
                                t[idx5 + 1] = a[r][c + 7];
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
                                a[r][c] = t[idx2];
                                a[r][c + 1] = t[idx2 + 1];
                                a[r][c + 2] = t[idx3];
                                a[r][c + 3] = t[idx3 + 1];
                                a[r][c + 4] = t[idx4];
                                a[r][c + 5] = t[idx4 + 1];
                                a[r][c + 6] = t[idx5];
                                a[r][c + 7] = t[idx5 + 1];
                            }
                        }
                    } else if (columns == 4 * nthreads) {
                        for (int r = 0; r < rows; r++) {
                            idx2 = 2 * r;
                            idx3 = 2 * rows + 2 * r;
                            t[idx2] = a[r][4 * n0];
                            t[idx2 + 1] = a[r][4 * n0 + 1];
                            t[idx3] = a[r][4 * n0 + 2];
                            t[idx3 + 1] = a[r][4 * n0 + 3];
                        }
                        fftRows.complexInverse(t, 0, scale);
                        fftRows.complexInverse(t, 2 * rows, scale);
                        for (int r = 0; r < rows; r++) {
                            idx2 = 2 * r;
                            idx3 = 2 * rows + 2 * r;
                            a[r][4 * n0] = t[idx2];
                            a[r][4 * n0 + 1] = t[idx2 + 1];
                            a[r][4 * n0 + 2] = t[idx3];
                            a[r][4 * n0 + 3] = t[idx3 + 1];
                        }
                    } else if (columns == 2 * nthreads) {
                        for (int r = 0; r < rows; r++) {
                            idx2 = 2 * r;
                            t[idx2] = a[r][2 * n0];
                            t[idx2 + 1] = a[r][2 * n0 + 1];
                        }
                        fftRows.complexInverse(t, 0, scale);
                        for (int r = 0; r < rows; r++) {
                            idx2 = 2 * r;
                            a[r][2 * n0] = t[idx2];
                            a[r][2 * n0 + 1] = t[idx2 + 1];
                        }
                    }
                }
            });
        }
        try {
            ConcurrencyUtils.waitForCompletion(futures);
        } catch (InterruptedException ex) {
            Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    private void fillSymmetric(final float[] a)
    {
        final int twon2 = 2 * columns;
        int idx1, idx2, idx3, idx4;
        int n1d2 = rows / 2;

        for (int r = (rows - 1); r >= 1; r--) {
            idx1 = r * columns;
            idx2 = 2 * idx1;
            for (int c = 0; c < columns; c += 2) {
                a[idx2 + c] = a[idx1 + c];
                a[idx1 + c] = 0;
                a[idx2 + c + 1] = a[idx1 + c + 1];
                a[idx1 + c + 1] = 0;
            }
        }
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && useThreads && (n1d2 >= nthreads)) {
            Future<?>[] futures = new Future[nthreads];
            int l1k = n1d2 / nthreads;
            final int newn2 = 2 * columns;
            for (int i = 0; i < nthreads; i++) {
                final int l1offa, l1stopa, l2offa, l2stopa;
                if (i == 0) {
                    l1offa = i * l1k + 1;
                } else {
                    l1offa = i * l1k;
                }
                l1stopa = i * l1k + l1k;
                l2offa = i * l1k;
                if (i == nthreads - 1) {
                    l2stopa = i * l1k + l1k + 1;
                } else {
                    l2stopa = i * l1k + l1k;
                }
                futures[i] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        int idx1, idx2, idx3, idx4;

                        for (int r = l1offa; r < l1stopa; r++) {
                            idx1 = r * newn2;
                            idx2 = (rows - r) * newn2;
                            idx3 = idx1 + columns;
                            a[idx3] = a[idx2 + 1];
                            a[idx3 + 1] = -a[idx2];
                        }
                        for (int r = l1offa; r < l1stopa; r++) {
                            idx1 = r * newn2;
                            idx3 = (rows - r + 1) * newn2;
                            for (int c = columns + 2; c < newn2; c += 2) {
                                idx2 = idx3 - c;
                                idx4 = idx1 + c;
                                a[idx4] = a[idx2];
                                a[idx4 + 1] = -a[idx2 + 1];

                            }
                        }
                        for (int r = l2offa; r < l2stopa; r++) {
                            idx3 = ((rows - r) % rows) * newn2;
                            idx4 = r * newn2;
                            for (int c = 0; c < newn2; c += 2) {
                                idx1 = idx3 + (newn2 - c) % newn2;
                                idx2 = idx4 + c;
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
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }

        } else {

            for (int r = 1; r < n1d2; r++) {
                idx2 = r * twon2;
                idx3 = (rows - r) * twon2;
                a[idx2 + columns] = a[idx3 + 1];
                a[idx2 + columns + 1] = -a[idx3];
            }

            for (int r = 1; r < n1d2; r++) {
                idx2 = r * twon2;
                idx3 = (rows - r + 1) * twon2;
                for (int c = columns + 2; c < twon2; c += 2) {
                    a[idx2 + c] = a[idx3 - c];
                    a[idx2 + c + 1] = -a[idx3 - c + 1];

                }
            }
            for (int r = 0; r <= rows / 2; r++) {
                idx1 = r * twon2;
                idx4 = ((rows - r) % rows) * twon2;
                for (int c = 0; c < twon2; c += 2) {
                    idx2 = idx1 + c;
                    idx3 = idx4 + (twon2 - c) % twon2;
                    a[idx3] = a[idx2];
                    a[idx3 + 1] = -a[idx2 + 1];
                }
            }
        }
        a[columns] = -a[1];
        a[1] = 0;
        idx1 = n1d2 * twon2;
        a[idx1 + columns] = -a[idx1 + 1];
        a[idx1 + 1] = 0;
        a[idx1 + columns + 1] = 0;
    }

    private void fillSymmetric(final FloatLargeArray a)
    {
        final long twon2 = 2 * columnsl;
        long idx1, idx2, idx3, idx4;
        long n1d2 = rowsl / 2;

        for (long r = (rowsl - 1); r >= 1; r--) {
            idx1 = r * columnsl;
            idx2 = 2 * idx1;
            for (long c = 0; c < columnsl; c += 2) {
                a.setDouble(idx2 + c, a.getFloat(idx1 + c));
                a.setDouble(idx1 + c, 0);
                a.setDouble(idx2 + c + 1, a.getFloat(idx1 + c + 1));
                a.setDouble(idx1 + c + 1, 0);
            }
        }
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && useThreads && (n1d2 >= nthreads)) {
            Future<?>[] futures = new Future[nthreads];
            long l1k = n1d2 / nthreads;
            final long newn2 = 2 * columnsl;
            for (int i = 0; i < nthreads; i++) {
                final long l1offa, l1stopa, l2offa, l2stopa;
                if (i == 0) {
                    l1offa = i * l1k + 1;
                } else {
                    l1offa = i * l1k;
                }
                l1stopa = i * l1k + l1k;
                l2offa = i * l1k;
                if (i == nthreads - 1) {
                    l2stopa = i * l1k + l1k + 1;
                } else {
                    l2stopa = i * l1k + l1k;
                }
                futures[i] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        long idx1, idx2, idx3, idx4;

                        for (long r = l1offa; r < l1stopa; r++) {
                            idx1 = r * newn2;
                            idx2 = (rowsl - r) * newn2;
                            idx3 = idx1 + columnsl;
                            a.setDouble(idx3, a.getFloat(idx2 + 1));
                            a.setDouble(idx3 + 1, -a.getFloat(idx2));
                        }
                        for (long r = l1offa; r < l1stopa; r++) {
                            idx1 = r * newn2;
                            idx3 = (rowsl - r + 1) * newn2;
                            for (long c = columnsl + 2; c < newn2; c += 2) {
                                idx2 = idx3 - c;
                                idx4 = idx1 + c;
                                a.setDouble(idx4, a.getFloat(idx2));
                                a.setDouble(idx4 + 1, -a.getFloat(idx2 + 1));

                            }
                        }
                        for (long r = l2offa; r < l2stopa; r++) {
                            idx3 = ((rowsl - r) % rowsl) * newn2;
                            idx4 = r * newn2;
                            for (long c = 0; c < newn2; c += 2) {
                                idx1 = idx3 + (newn2 - c) % newn2;
                                idx2 = idx4 + c;
                                a.setDouble(idx1, a.getFloat(idx2));
                                a.setDouble(idx1 + 1, -a.getFloat(idx2 + 1));
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }

        } else {

            for (long r = 1; r < n1d2; r++) {
                idx2 = r * twon2;
                idx3 = (rowsl - r) * twon2;
                a.setDouble(idx2 + columnsl, a.getFloat(idx3 + 1));
                a.setDouble(idx2 + columnsl + 1, -a.getFloat(idx3));
            }

            for (long r = 1; r < n1d2; r++) {
                idx2 = r * twon2;
                idx3 = (rowsl - r + 1) * twon2;
                for (long c = columnsl + 2; c < twon2; c += 2) {
                    a.setDouble(idx2 + c, a.getFloat(idx3 - c));
                    a.setDouble(idx2 + c + 1, -a.getFloat(idx3 - c + 1));

                }
            }
            for (long r = 0; r <= rowsl / 2; r++) {
                idx1 = r * twon2;
                idx4 = ((rowsl - r) % rowsl) * twon2;
                for (long c = 0; c < twon2; c += 2) {
                    idx2 = idx1 + c;
                    idx3 = idx4 + (twon2 - c) % twon2;
                    a.setDouble(idx3, a.getFloat(idx2));
                    a.setDouble(idx3 + 1, -a.getFloat(idx2 + 1));
                }
            }
        }
        a.setDouble(columnsl, -a.getFloat(1));
        a.setDouble(1, 0);
        idx1 = n1d2 * twon2;
        a.setDouble(idx1 + columnsl, -a.getFloat(idx1 + 1));
        a.setDouble(idx1 + 1, 0);
        a.setDouble(idx1 + columnsl + 1, 0);
    }

    private void fillSymmetric(final float[][] a)
    {
        final int newn2 = 2 * columns;
        int n1d2 = rows / 2;

        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && useThreads && (n1d2 >= nthreads)) {
            Future<?>[] futures = new Future[nthreads];
            int l1k = n1d2 / nthreads;
            for (int i = 0; i < nthreads; i++) {
                final int l1offa, l1stopa, l2offa, l2stopa;
                if (i == 0) {
                    l1offa = i * l1k + 1;
                } else {
                    l1offa = i * l1k;
                }
                l1stopa = i * l1k + l1k;
                l2offa = i * l1k;
                if (i == nthreads - 1) {
                    l2stopa = i * l1k + l1k + 1;
                } else {
                    l2stopa = i * l1k + l1k;
                }
                futures[i] = ConcurrencyUtils.submit(new Runnable()
                {
                    public void run()
                    {
                        int idx1, idx2;
                        for (int r = l1offa; r < l1stopa; r++) {
                            idx1 = rows - r;
                            a[r][columns] = a[idx1][1];
                            a[r][columns + 1] = -a[idx1][0];
                        }
                        for (int r = l1offa; r < l1stopa; r++) {
                            idx1 = rows - r;
                            for (int c = columns + 2; c < newn2; c += 2) {
                                idx2 = newn2 - c;
                                a[r][c] = a[idx1][idx2];
                                a[r][c + 1] = -a[idx1][idx2 + 1];

                            }
                        }
                        for (int r = l2offa; r < l2stopa; r++) {
                            idx1 = (rows - r) % rows;
                            for (int c = 0; c < newn2; c = c + 2) {
                                idx2 = (newn2 - c) % newn2;
                                a[idx1][idx2] = a[r][c];
                                a[idx1][idx2 + 1] = -a[r][c + 1];
                            }
                        }
                    }
                });
            }
            try {
                ConcurrencyUtils.waitForCompletion(futures);
            } catch (InterruptedException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            } catch (ExecutionException ex) {
                Logger.getLogger(FloatFFT_2D.class.getName()).log(Level.SEVERE, null, ex);
            }
        } else {

            for (int r = 1; r < n1d2; r++) {
                int idx1 = rows - r;
                a[r][columns] = a[idx1][1];
                a[r][columns + 1] = -a[idx1][0];
            }
            for (int r = 1; r < n1d2; r++) {
                int idx1 = rows - r;
                for (int c = columns + 2; c < newn2; c += 2) {
                    int idx2 = newn2 - c;
                    a[r][c] = a[idx1][idx2];
                    a[r][c + 1] = -a[idx1][idx2 + 1];
                }
            }
            for (int r = 0; r <= rows / 2; r++) {
                int idx1 = (rows - r) % rows;
                for (int c = 0; c < newn2; c += 2) {
                    int idx2 = (newn2 - c) % newn2;
                    a[idx1][idx2] = a[r][c];
                    a[idx1][idx2 + 1] = -a[r][c + 1];
                }
            }
        }
        a[0][columns] = -a[0][1];
        a[0][1] = 0;
        a[n1d2][columns] = -a[n1d2][1];
        a[n1d2][1] = 0;
        a[n1d2][columns + 1] = 0;
    }
}
