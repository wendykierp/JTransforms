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

// @formatter:off
import pl.edu.icm.jlargearrays.DoubleLargeArray;
import pl.edu.icm.jlargearrays.FloatLargeArray;
import static org.apache.commons.math3.util.FastMath.*;

/**
 *
 * This is a set of utility methods for R/W access to data resulting from a call
 * to the Fourier transform of <em>real</em> data. Memory optimized methods,
 * namely
 * <ul>
 * <li>{@link DoubleFFT_3D#realForward(double[])}</li>
 * <li>{@link DoubleFFT_3D#realForward(DoubleLargeArray)}</li>
 * <li>{@link DoubleFFT_3D#realForward(double[][][])}</li>
 * <li>{@link FloatFFT_3D#realForward(float[])}</li>
 * <li>{@link FloatFFT_3D#realForward(FloatLargeArray)}</li>
 * <li>{@link FloatFFT_3D#realForward(float[][][])}</li>
 * </ul>
 * are implemented to handle this case specifically. However, packing of the
 * data in the data array is somewhat obscure. This class provides methods for
 * direct access to the data, without the burden of all necessary tests.
 * <h3>Example for Fourier Transform of real, double precision 1d data</h3>
 *  
 * <pre>
 * DoubleFFT_3D fft = new DoubleFFT_2D(slices, rows, columns);
 * double[] data = new double[2 * slices * rows * columns];
 * ...
 * fft.realForwardFull(data);
 * data[(s1 * rows + r1) * 2 * columns + c1] = val1;
 * val2 = data[(s2 * rows + r2) * 2 * columns + c2];
 * </pre>
 * is equivalent to
 * <pre>
 *   DoubleFFT_3D fft = new DoubleFFT_3D(slices, rows, columns);
 *   RealFFTUtils_3D unpacker = new RealFFTUtils_3D(slices, rows, columns);
 *   double[] data = new double[slices * rows * columns];
 *   ...
 *   fft.realForward(data);
 *   unpacker.pack(val1, s1, r1, c1, data);
 *   val2 = unpacker.unpack(s2, r2, c2, data, 0);
 * </pre>
 * Even (resp. odd) values of <code>c</code> correspond to the real (resp.
 * imaginary) part of the Fourier mode.
 * <h3>Example for Fourier Transform of real, double precision 3d data</h3>
 *  
 * <pre>
 * DoubleFFT_3D fft = new DoubleFFT_3D(slices, rows, columns);
 * double[][][] data = new double[slices][rows][2 * columns];
 * ...
 * fft.realForwardFull(data);
 * data[s1][r1][c1] = val1;
 * val2 = data[s2][r2][c2];
 * </pre>
 * is equivalent to
 * <pre>
 *   DoubleFFT_3D fft = new DoubleFFT_3D(slices, rows, columns);
 *   RealFFTUtils_3D unpacker = new RealFFTUtils_3D(slices, rows, columns);
 *   double[][][] data = new double[slices][rows][columns];
 *   ...
 *   fft.realForward(data);
 *   unpacker.pack(val1, s1, r1, c1, data);
 *   val2 = unpacker.unpack(s2, r2, c2, data, 0);
 * </pre>
 * Even (resp. odd) values of <code>c</code> correspond to the real (resp.
 * imaginary) part of the Fourier mode.
 *  
 * @author S&eacute;bastien Brisard
 */
// @formatter:on
public class RealFFTUtils_3D
{

    /**
     * The constant <code>int</code> value of 1.
     */
    private static final int ONE = 1;

    /**
     * The constant <code>int</code> value of 2.
     */
    private static final int TWO = 2;

    /**
     * The constant <code>int</code> value of 0.
     */
    private static final int ZERO = 0;

    /**
     * The constant <code>int</code> value of 1.
     */
    private static final long ONEL = 1;

    /**
     * The constant <code>int</code> value of 2.
     */
    private static final long TWOL = 2;

    /**
     * The constant <code>int</code> value of 0.
     */
    private static final long ZEROL = 0;

    /**
     * The size of the data in the third direction.
     */
    private final int columns;

    /**
     * The size of the data in the third direction.
     */
    private final long columnsl;

    /**
     * The size of the data in the second direction.
     */
    private final int rows;

    /**
     * The size of the data in the second direction.
     */
    private final long rowsl;

    /**
     * The constant value of <code>2 * columns</code>.
     */
    private final int rowStride;

    /**
     * The constant value of <code>2 * columns</code>.
     */
    private final long rowStridel;

    /**
     * The size of the data in the first direction.
     */
    private final int slices;

    /**
     * The size of the data in the first direction.
     */
    private final long slicesl;

    /**
     * The constant value of <code>2 * rows * columns</code>.
     */
    private final int sliceStride;

    /**
     * The constant value of <code>2 * rows * columns</code>.
     */
    private final long sliceStridel;

    /**
     * Creates a new instance of this class. The size of the underlying
     * {@link DoubleFFT_3D} or {@link FloatFFT_3D} must be specified.
     *  
     * @param slices
     *                number of slices
     * @param rows
     *                number of rows
     * @param columns
     *                number of columns
     */
    public RealFFTUtils_3D(final long slices, final long rows, final long columns)
    {
        this.slices = (int) slices;
        this.rows = (int) rows;
        this.columns = (int) columns;
        this.rowStride = (int) columns;
        this.sliceStride = (int) rows * (int) this.rowStride;
        this.slicesl = slices;
        this.rowsl = rows;
        this.columnsl = columns;
        this.rowStridel = columns;
        this.sliceStridel = rows * this.rowStridel;
    }

    /**
     *
     * Returns the 1d index of the specified 3d Fourier mode. In other words, if
     * <code>packed</code> contains the transformed data following a call to
     * {@link DoubleFFT_3D#realForwardFull(double[])} or
     * {@link FloatFFT_3D#realForward(float[])}, then the returned value
     * <code>index</code> gives access to the <code>[s][r][c]</code> Fourier
     * mode
     * <ul>
     * <li>if <code>index == {@link Integer#MIN_VALUE}</code>, then the Fourier
     * mode is zero,</li>
     * <li>if <code>index &ge; 0</code>, then the Fourier mode is
     * <code>packed[index]</code>,</li>
     * <li>if <code>index &lt; 0</code>, then the Fourier mode is
     * <code>-packed[-index]</code>,</li>
     * </ul>
     *  
     * @param s
     *          the slice index
     * @param r
     *          the row index
     * @param c
     *          the column index
     *  
     * @return the value of <code>index</code>
     */
    public int getIndex(final int s, final int r, final int c)
    {
        final int cmod2 = c & ONE;
        final int rmul2 = r << ONE;
        final int smul2 = s << ONE;
        final int ss = s == ZERO ? ZERO : slices - s;
        final int rr = r == ZERO ? ZERO : rows - r;
        if (c <= ONE) {
            if (r == ZERO) {
                if (s == ZERO) {
                    return c == ZERO ? ZERO : Integer.MIN_VALUE;
                } else if (smul2 < slices) {
                    return s * sliceStride + c;
                } else if (smul2 > slices) {
                    final int index = ss * sliceStride;
                    return cmod2 == ZERO ? index : -(index + ONE);
                } else {
                    return cmod2 == ZERO ? s * sliceStride : Integer.MIN_VALUE;
                }
            } else if (rmul2 < rows) {
                return s * sliceStride + r * rowStride + c;
            } else if (rmul2 > rows) {
                final int index = ss * sliceStride + rr * rowStride;
                return cmod2 == ZERO ? index : -(index + ONE);
            } else if (s == ZERO) {
                return cmod2 == ZERO ? r * rowStride : Integer.MIN_VALUE;
            } else if (smul2 < slices) {
                return s * sliceStride + r * rowStride + c;
            } else if (smul2 > slices) {
                final int index = ss * sliceStride + r * rowStride;
                return cmod2 == ZERO ? index : -(index + ONE);
            } else {
                final int index = s * sliceStride + r * rowStride;
                return cmod2 == ZERO ? index : Integer.MIN_VALUE;
            }
        } else if (c < columns) {
            return s * sliceStride + r * rowStride + c;
        } else if (c > columns + ONE) {
            final int cc = (columns << ONE) - c;
            final int index = ss * sliceStride + rr * rowStride + cc;
            return cmod2 == ZERO ? index : -(index + TWO);
        } else if (r == ZERO) {
            if (s == ZERO) {
                return cmod2 == ZERO ? ONE : Integer.MIN_VALUE;
            } else if (smul2 < slices) {
                final int index = ss * sliceStride;
                return cmod2 == ZERO ? index + ONE : -index;
            } else if (smul2 > slices) {
                final int index = s * sliceStride;
                return cmod2 == ZERO ? index + ONE : index;
            } else {
                final int index = s * sliceStride;
                return cmod2 == ZERO ? index + ONE : Integer.MIN_VALUE;
            }
        } else if (rmul2 < rows) {
            final int index = ss * sliceStride + rr * rowStride;
            return cmod2 == ZERO ? index + ONE : -index;
        } else if (rmul2 > rows) {
            final int index = s * sliceStride + r * rowStride;
            return cmod2 == ZERO ? index + ONE : index;
        } else if (s == ZERO) {
            final int index = r * rowStride + ONE;
            return cmod2 == ZERO ? index : Integer.MIN_VALUE;
        } else if (smul2 < slices) {
            final int index = ss * sliceStride + r * rowStride;
            return cmod2 == ZERO ? index + ONE : -index;
        } else if (smul2 > slices) {
            final int index = s * sliceStride + r * rowStride;
            return cmod2 == ZERO ? index + ONE : index;
        } else {
            final int index = s * sliceStride + r * rowStride;
            return cmod2 == ZERO ? index + ONE : Integer.MIN_VALUE;
        }
    }

    /**
     *
     * Returns the 1d index of the specified 3d Fourier mode. In other words, if
     * <code>packed</code> contains the transformed data following a call to
     * {@link DoubleFFT_3D#realForwardFull(double[])} or
     * {@link FloatFFT_3D#realForward(float[])}, then the returned value
     * <code>index</code> gives access to the <code>[s][r][c]</code> Fourier
     * mode
     * <ul>
     * <li>if <code>index == {@link Integer#MIN_VALUE}</code>, then the Fourier
     * mode is zero,</li>
     * <li>if <code>index &ge; 0</code>, then the Fourier mode is
     * <code>packed[index]</code>,</li>
     * <li>if <code>index &lt; 0</code>, then the Fourier mode is
     * <code>-packed[-index]</code>,</li>
     * </ul>
     *  
     * @param s
     *          the slice index
     * @param r
     *          the row index
     * @param c
     *          the column index
     *  
     * @return the value of <code>index</code>
     */
    public long getIndex(final long s, final long r, final long c)
    {
        final long cmod2 = c & ONEL;
        final long rmul2 = r << ONEL;
        final long smul2 = s << ONEL;
        final long ss = s == ZEROL ? ZEROL : slicesl - s;
        final long rr = r == ZEROL ? ZEROL : rowsl - r;
        if (c <= ONEL) {
            if (r == ZEROL) {
                if (s == ZEROL) {
                    return c == ZEROL ? ZEROL : Long.MIN_VALUE;
                } else if (smul2 < slicesl) {
                    return s * sliceStridel + c;
                } else if (smul2 > slicesl) {
                    final long index = ss * sliceStridel;
                    return cmod2 == ZEROL ? index : -(index + ONEL);
                } else {
                    return cmod2 == ZEROL ? s * sliceStridel : Long.MIN_VALUE;
                }
            } else if (rmul2 < rowsl) {
                return s * sliceStridel + r * rowStridel + c;
            } else if (rmul2 > rowsl) {
                final long index = ss * sliceStridel + rr * rowStridel;
                return cmod2 == ZEROL ? index : -(index + ONEL);
            } else if (s == ZEROL) {
                return cmod2 == ZEROL ? r * rowStridel : Long.MIN_VALUE;
            } else if (smul2 < slicesl) {
                return s * sliceStridel + r * rowStridel + c;
            } else if (smul2 > slicesl) {
                final long index = ss * sliceStridel + r * rowStridel;
                return cmod2 == ZEROL ? index : -(index + ONEL);
            } else {
                final long index = s * sliceStridel + r * rowStridel;
                return cmod2 == ZEROL ? index : Long.MIN_VALUE;
            }
        } else if (c < columnsl) {
            return s * sliceStridel + r * rowStridel + c;
        } else if (c > columnsl + ONEL) {
            final long cc = (columnsl << ONEL) - c;
            final long index = ss * sliceStridel + rr * rowStridel + cc;
            return cmod2 == ZEROL ? index : -(index + TWO);
        } else if (r == ZEROL) {
            if (s == ZEROL) {
                return cmod2 == ZEROL ? ONEL : Long.MIN_VALUE;
            } else if (smul2 < slicesl) {
                final long index = ss * sliceStridel;
                return cmod2 == ZEROL ? index + ONEL : -index;
            } else if (smul2 > slicesl) {
                final long index = s * sliceStridel;
                return cmod2 == ZEROL ? index + ONEL : index;
            } else {
                final long index = s * sliceStridel;
                return cmod2 == ZEROL ? index + ONEL : Long.MIN_VALUE;
            }
        } else if (rmul2 < rowsl) {
            final long index = ss * sliceStridel + rr * rowStridel;
            return cmod2 == ZEROL ? index + ONEL : -index;
        } else if (rmul2 > rowsl) {
            final long index = s * sliceStridel + r * rowStridel;
            return cmod2 == ZEROL ? index + ONEL : index;
        } else if (s == ZEROL) {
            final long index = r * rowStridel + ONEL;
            return cmod2 == ZEROL ? index : Long.MIN_VALUE;
        } else if (smul2 < slicesl) {
            final long index = ss * sliceStridel + r * rowStridel;
            return cmod2 == ZEROL ? index + ONEL : -index;
        } else if (smul2 > slicesl) {
            final long index = s * sliceStridel + r * rowStridel;
            return cmod2 == ZEROL ? index + ONEL : index;
        } else {
            final long index = s * sliceStridel + r * rowStridel;
            return cmod2 == ZEROL ? index + ONEL : Long.MIN_VALUE;
        }
    }

    /**
     * Sets the specified Fourier mode of the transformed data. The data array
     * results from a call to {@link DoubleFFT_3D#realForward(double[])}.
     *  
     * @param val
     *               the new value of the <code>[s][r][c]</code> Fourier mode
     * @param s
     *               the slice index
     * @param r
     *               the row index
     * @param c
     *               the column index
     * @param packed
     *               the transformed data
     * @param pos
     *               index of the first element in array <code>packed</code>
     */
    public void pack(final double val, final int s, final int r, final int c,
                     final double[] packed, final int pos)
    {
        final int i = getIndex(s, r, c);
        if (i >= 0) {
            packed[pos + i] = val;
        } else if (i > Integer.MIN_VALUE) {
            packed[pos - i] = -val;
        } else {
            throw new IllegalArgumentException(String.format(
                "[%d][%d][%d] component cannot be modified (always zero)",
                s, r, c));
        }
    }

    /**
     * Sets the specified Fourier mode of the transformed data. The data array
     * results from a call to {@link DoubleFFT_3D#realForward(DoubleLargeArray)}.
     *  
     * @param val
     *               the new value of the <code>[s][r][c]</code> Fourier mode
     * @param s
     *               the slice index
     * @param r
     *               the row index
     * @param c
     *               the column index
     * @param packed
     *               the transformed data
     * @param pos
     *               index of the first element in array <code>packed</code>
     */
    public void pack(final double val, final long s, final long r, final long c,
                     final DoubleLargeArray packed, final long pos)
    {
        final long i = getIndex(s, r, c);
        if (i >= 0) {
            packed.setDouble(pos + i, val);
        } else if (i > Long.MIN_VALUE) {
            packed.setDouble(pos - i, -val);
        } else {
            throw new IllegalArgumentException(String.format(
                "[%d][%d][%d] component cannot be modified (always zero)",
                s, r, c));
        }
    }

    /**
     * Sets the specified Fourier mode of the transformed data. The data array
     * results from a call to {@link DoubleFFT_3D#realForward(double[][][])}.
     *  
     * @param val
     *               the new value of the <code>[s][r][c]</code> Fourier mode
     * @param s
     *               the slice index
     * @param r
     *               the row index
     * @param c
     *               the column index
     * @param packed
     *               the transformed data
     */
    public void pack(final double val, final int s, final int r, final int c,
                     final double[][][] packed)
    {
        final int i = getIndex(s, r, c);
        final int ii = abs(i);
        final int ss = ii / sliceStride;
        final int remainder = ii % sliceStride;
        final int rr = remainder / rowStride;
        final int cc = remainder % rowStride;
        if (i >= 0) {
            packed[ss][rr][cc] = val;
        } else if (i > Integer.MIN_VALUE) {
            packed[ss][rr][cc] = -val;
        } else {
            throw new IllegalArgumentException(
                String.format(
                    "[%d][%d] component cannot be modified (always zero)",
                    r, c));
        }
    }

    /**
     * Sets the specified Fourier mode of the transformed data. The data array
     * results from a call to {@link FloatFFT_3D#realForward(float[])}.
     *  
     * @param val
     *               the new value of the <code>[s][r][c]</code> Fourier mode
     * @param s
     *               the slice index
     * @param r
     *               the row index
     * @param c
     *               the column index
     * @param packed
     *               the transformed data
     * @param pos
     *               index of the first element in array <code>packed</code>
     */
    public void pack(final float val, final int s, final int r, final int c,
                     final float[] packed, final int pos)
    {
        final int i = getIndex(s, r, c);
        if (i >= 0) {
            packed[pos + i] = val;
        } else if (i > Integer.MIN_VALUE) {
            packed[pos - i] = -val;
        } else {
            throw new IllegalArgumentException(String.format(
                "[%d][%d][%d] component cannot be modified (always zero)",
                s, r, c));
        }
    }

    /**
     * Sets the specified Fourier mode of the transformed data. The data array
     * results from a call to {@link FloatFFT_3D#realForward(FloatLargeArray)}.
     *  
     * @param val
     *               the new value of the <code>[s][r][c]</code> Fourier mode
     * @param s
     *               the slice index
     * @param r
     *               the row index
     * @param c
     *               the column index
     * @param packed
     *               the transformed data
     * @param pos
     *               index of the first element in array <code>packed</code>
     */
    public void pack(final float val, final long s, final long r, final long c,
                     final FloatLargeArray packed, final long pos)
    {
        final long i = getIndex(s, r, c);
        if (i >= 0) {
            packed.setFloat(pos + i, val);
        } else if (i > Long.MIN_VALUE) {
            packed.setFloat(pos - i, -val);
        } else {
            throw new IllegalArgumentException(String.format(
                "[%d][%d][%d] component cannot be modified (always zero)",
                s, r, c));
        }
    }

    /**
     * Sets the specified Fourier mode of the transformed data. The data array
     * results from a call to {@link FloatFFT_3D#realForward(float[][][])}.
     *  
     * @param val
     *               the new value of the <code>[s][r][c]</code> Fourier mode
     * @param s
     *               the slice index
     * @param r
     *               the row index
     * @param c
     *               the column index
     * @param packed
     *               the transformed data
     */
    public void pack(final float val, final int s, final int r, final int c,
                     final float[][][] packed)
    {
        final int i = getIndex(s, r, c);
        final int ii = abs(i);
        final int ss = ii / sliceStride;
        final int remainder = ii % sliceStride;
        final int rr = remainder / rowStride;
        final int cc = remainder % rowStride;
        if (i >= 0) {
            packed[ss][rr][cc] = val;
        } else if (i > Integer.MIN_VALUE) {
            packed[ss][rr][cc] = -val;
        } else {
            throw new IllegalArgumentException(String.format(
                "[%d][%d][%d] component cannot be modified (always zero)",
                s, r, c));
        }
    }

    /**
     * Returns the specified Fourier mode of the transformed data. The data
     * array results from a call to {@link DoubleFFT_3D#realForward(double[])}.
     *  
     * @param s
     *               the slice index
     * @param r
     *               the row index
     * @param c
     *               the column index
     * @param packed
     *               the transformed data
     * @param pos
     *               index of the first element in array <code>packed</code>
     *  
     * @return the value of the <code>[s][r][c]</code> Fourier mode
     */
    public double unpack(final int s, final int r, final int c,
                         final double[] packed, final int pos)
    {
        final int i = getIndex(s, r, c);
        if (i >= 0) {
            return packed[pos + i];
        } else if (i > Integer.MIN_VALUE) {
            return -packed[pos - i];
        } else {
            return ZERO;
        }
    }

    /**
     * Returns the specified Fourier mode of the transformed data. The data
     * array results from a call to {@link DoubleFFT_3D#realForward(DoubleLargeArray)}.
     *  
     * @param s
     *               the slice index
     * @param r
     *               the row index
     * @param c
     *               the column index
     * @param packed
     *               the transformed data
     * @param pos
     *               index of the first element in array <code>packed</code>
     *  
     * @return the value of the <code>[s][r][c]</code> Fourier mode
     */
    public double unpack(final long s, final long r, final long c,
                         final DoubleLargeArray packed, final long pos)
    {
        final long i = getIndex(s, r, c);
        if (i >= 0) {
            return packed.getDouble(pos + i);
        } else if (i > Long.MIN_VALUE) {
            return -packed.getDouble(pos - i);
        } else {
            return ZEROL;
        }
    }

    /**
     * Returns the specified Fourier mode of the transformed data. The data
     * array results from a call to
     * {@link DoubleFFT_3D#realForward(double[][][])} .
     *  
     * @param s
     *               the slice index
     * @param r
     *               the row index
     * @param c
     *               the column index
     * @param packed
     *               the transformed data
     *  
     * @return the value of the <code>[s][r][c]</code> Fourier mode
     */
    public double unpack(final int s, final int r, final int c,
                         final double[][][] packed)
    {
        final int i = getIndex(s, r, c);
        final int ii = abs(i);
        final int ss = ii / sliceStride;
        final int remainder = ii % sliceStride;
        final int rr = remainder / rowStride;
        final int cc = remainder % rowStride;
        if (i >= 0) {
            return packed[ss][rr][cc];
        } else if (i > Integer.MIN_VALUE) {
            return -packed[ss][rr][cc];
        } else {
            return ZERO;
        }
    }

    /**
     * Returns the specified Fourier mode of the transformed data. The data
     * array results from a call to {@link FloatFFT_3D#realForward(float[])} .
     *  
     * @param s
     *               the slice index
     * @param r
     *               the row index
     * @param c
     *               the column index
     * @param packed
     *               the transformed data
     * @param pos
     *               index of the first element in array <code>packed</code>
     *  
     * @return the value of the <code>[s][r][c]</code> Fourier mode
     */
    public float unpack(final int s, final int r, final int c,
                        final float[] packed, final int pos)
    {
        final int i = getIndex(s, r, c);
        if (i >= 0) {
            return packed[pos + i];
        } else if (i > Integer.MIN_VALUE) {
            return -packed[pos - i];
        } else {
            return ZERO;
        }
    }

    /**
     * Returns the specified Fourier mode of the transformed data. The data
     * array results from a call to {@link FloatFFT_3D#realForward(FloatLargeArray)} .
     *  
     * @param s
     *               the slice index
     * @param r
     *               the row index
     * @param c
     *               the column index
     * @param packed
     *               the transformed data
     * @param pos
     *               index of the first element in array <code>packed</code>
     *  
     * @return the value of the <code>[s][r][c]</code> Fourier mode
     */
    public float unpack(final long s, final long r, final long c,
                        final FloatLargeArray packed, final long pos)
    {
        final long i = getIndex(s, r, c);
        if (i >= 0) {
            return packed.getFloat(pos + i);
        } else if (i > Long.MIN_VALUE) {
            return -packed.getFloat(pos - i);
        } else {
            return ZEROL;
        }
    }

    /**
     * Returns the specified Fourier mode of the transformed data. The data
     * array results from a call to {@link FloatFFT_3D#realForward(float[][][])}
     * .
     *  
     * @param s
     *               the slice index
     * @param r
     *               the row index
     * @param c
     *               the column index
     * @param packed
     *               the transformed data
     *  
     * @return the value of the <code>[s][r][c]</code> Fourier mode
     */
    public float unpack(final int s, final int r, final int c,
                        final float[][][] packed)
    {
        final int i = getIndex(s, r, c);
        final int ii = abs(i);
        final int ss = ii / sliceStride;
        final int remainder = ii % sliceStride;
        final int rr = remainder / rowStride;
        final int cc = remainder % rowStride;
        if (i >= 0) {
            return packed[ss][rr][cc];
        } else if (i > Integer.MIN_VALUE) {
            return -packed[ss][rr][cc];
        } else {
            return ZERO;
        }
    }
}
