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

/**
 *
 * This is a set of utility methods for R/W access to data resulting from a call
 * to the Fourier transform of <em>real</em> data. Memory optimized methods,
 * namely
 * <ul>
 * <li>{@link DoubleFFT_2D#realForward(double[])}</li>
 * <li>{@link DoubleFFT_2D#realForward(DoubleLargeArray)}</li>
 * <li>{@link DoubleFFT_2D#realForward(double[][])}</li>
 * <li>{@link FloatFFT_2D#realForward(float[])}</li>
 * <li>{@link FloatFFT_2D#realForward(FloatLargeArray)}</li>
 * <li>{@link FloatFFT_2D#realForward(float[][])}</li>
 * </ul>
 * are implemented to handle this case specifically. However, packing of the
 * data in the data array is somewhat obscure. This class provides methods for
 * direct access to the data, without the burden of all necessary tests.
 * <h3>Example for Fourier Transform of real, double precision 1d data</h3>
 *  
 * <pre>
 * DoubleFFT_2D fft = new DoubleFFT_2D(rows, columns);
 * double[] data = new double[2 * rows * columns];
 * ...
 * fft.realForwardFull(data);
 * data[r1 * 2 * columns + c1] = val1;
 * val2 = data[r2 * 2 * columns + c2];
 * </pre>
 * is equivalent to
 * <pre>
 *   DoubleFFT_2D fft = new DoubleFFT_2D(rows, columns);
 *   RealFFTUtils_2D unpacker = new RealFFTUtils_2D(rows, columns);
 *   double[] data = new double[rows * columns];
 *   ...
 *   fft.realForward(data);
 *   unpacker.pack(val1, r1, c1, data);
 *   val2 = unpacker.unpack(r2, c2, data, 0);
 * </pre>
 * Even (resp. odd) values of <code>c</code> correspond to the real (resp.
 * imaginary) part of the Fourier mode.
 * <h3>Example for Fourier Transform of real, double precision 2d data</h3>
 *  
 * <pre>
 * DoubleFFT_2D fft = new DoubleFFT_2D(rows, columns);
 * double[][] data = new double[rows][2 * columns];
 * ...
 * fft.realForwardFull(data);
 * data[r1][c1] = val1;
 * val2 = data[r2][c2];
 * </pre>
 * is equivalent to
 * <pre>
 *   DoubleFFT_2D fft = new DoubleFFT_2D(rows, columns);
 *   RealFFTUtils_2D unpacker = new RealFFTUtils_2D(rows, columns);
 *   double[][] data = new double[rows][columns];
 *   ...
 *   fft.realForward(data);
 *   unpacker.pack(val1, r1, c1, data);
 *   val2 = unpacker.unpack(r2, c2, data, 0);
 * </pre>
 * Even (resp. odd) values of <code>c</code> correspond to the real (resp.
 * imaginary) part of the Fourier mode.
 *  
 * @author S&eacute;bastien Brisard
 */
// @formatter:on
public class RealFFTUtils_2D
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
     * The size of the data in the second direction.
     */
    private final int columns;

    /**
     * The size of the data in the first direction.
     */
    private final int rows;

    /**
     * The size of the data in the second direction.
     */
    private final long columnsl;

    /**
     * The size of the data in the first direction.
     */
    private final long rowsl;

    /**
     * Creates a new instance of this class. The size of the underlying
     * {@link DoubleFFT_2D} or {@link FloatFFT_2D} must be specified.
     *  
     * @param rows
     *                number of rows
     * @param columns
     *                number of columns
     */
    public RealFFTUtils_2D(final long rows, final long columns)
    {
        this.columns = (int) columns;
        this.rows = (int) rows;
        this.columnsl = columns;
        this.rowsl = rows;

    }

    /**
     *
     * Returns the 1d index of the specified 2d Fourier mode. In other words, if
     * <code>packed</code> contains the transformed data following a call to
     * {@link DoubleFFT_2D#realForward(double[])} or
     * {@link FloatFFT_2D#realForward(float[])}, then the returned value
     * <code>index</code> gives access to the <code>[r][c]</code> Fourier mode
     * <ul>
     * <li>if <code>index == {@link Integer#MIN_VALUE}</code>, then the Fourier
     * mode is zero,</li>
     * <li>if <code>index &ge; 0</code>, then the Fourier mode is
     * <code>packed[index]</code>,</li>
     * <li>if <code>index &lt; 0 </code>, then the Fourier mode is
     * <code>-packed[-index]</code>,</li>
     * </ul>
     *  
     * @param r
     *          the row index
     * @param c
     *          the column index
     *  
     * @return the value of <code>index</code>
     */
    public int getIndex(final int r, final int c)
    {
        final int cmod2 = c & ONE;
        final int rmul2 = r << ONE;
        if (r != ZERO) {
            if (c <= ONE) {
                if (rmul2 == rows) {
                    if (cmod2 == ONE) {
                        return Integer.MIN_VALUE;
                    }
                    return ((rows * columns) >> ONE);
                } else if (rmul2 < rows) {
                    return columns * r + cmod2;
                } else if (cmod2 == ZERO) {
                    return columns * (rows - r);
                } else {
                    return -(columns * (rows - r) + ONE);
                }
            } else if ((c == columns) || (c == columns + ONE)) {
                if (rmul2 == rows) {
                    if (cmod2 == ONE) {
                        return Integer.MIN_VALUE;
                    }
                    return ((rows * columns) >> ONE) + ONE;
                } else if (rmul2 < rows) {
                    if (cmod2 == ZERO) {
                        return columns * (rows - r) + ONE;
                    } else {
                        return -(columns * (rows - r));
                    }
                } else {
                    return columns * r + ONE - cmod2;
                }
            } else if (c < columns) {
                return columns * r + c;
            } else if (cmod2 == ZERO) {
                return columns * (rows + TWO - r) - c;
            } else {
                return -(columns * (rows + TWO - r) - c + TWO);
            }
        } else if ((c == ONE) || (c == columns + ONE)) {
            return Integer.MIN_VALUE;
        } else if (c == columns) {
            return ONE;
        } else if (c < columns) {
            return c;
        } else if (cmod2 == ZERO) {
            return (columns << ONE) - c;
        } else {
            return -((columns << ONE) - c + TWO);
        }
    }

    /**
     *
     * Returns the 1d index of the specified 2d Fourier mode. In other words, if
     * <code>packed</code> contains the transformed data following a call to
     * {@link DoubleFFT_2D#realForward(DoubleLargeArray)} or
     * {@link FloatFFT_2D#realForward(FloatLargeArray)}, then the returned value
     * <code>index</code> gives access to the <code>[r][c]</code> Fourier mode
     * <ul>
     * <li>if <code>index == {@link Long#MIN_VALUE}</code>, then the Fourier
     * mode is zero,</li>
     * <li>if <code>index &ge; 0</code>, then the Fourier mode is
     * <code>packed[index]</code>,</li>
     * <li>if <code>index &lt; 0</code>, then the Fourier mode is
     * <code>-packed[-index]</code>,</li>
     * </ul>
     *  
     * @param r
     *          the row index
     * @param c
     *          the column index
     *  
     * @return the value of <code>index</code>
     */
    public long getIndex(final long r, final long c)
    {
        final long cmod2 = c & ONEL;
        final long rmul2 = r << ONEL;
        if (r != ZERO) {
            if (c <= ONEL) {
                if (rmul2 == rowsl) {
                    if (cmod2 == ONEL) {
                        return Long.MIN_VALUE;
                    }
                    return ((rowsl * columnsl) >> ONEL);
                } else if (rmul2 < rowsl) {
                    return columnsl * r + cmod2;
                } else if (cmod2 == ZEROL) {
                    return columnsl * (rowsl - r);
                } else {
                    return -(columnsl * (rowsl - r) + ONEL);
                }
            } else if ((c == columnsl) || (c == columnsl + ONEL)) {
                if (rmul2 == rowsl) {
                    if (cmod2 == ONEL) {
                        return Long.MIN_VALUE;
                    }
                    return ((rowsl * columnsl) >> ONEL) + ONEL;
                } else if (rmul2 < rowsl) {
                    if (cmod2 == ZEROL) {
                        return columnsl * (rowsl - r) + ONEL;
                    } else {
                        return -(columnsl * (rowsl - r));
                    }
                } else {
                    return columnsl * r + ONEL - cmod2;
                }
            } else if (c < columnsl) {
                return columnsl * r + c;
            } else if (cmod2 == ZEROL) {
                return columnsl * (rowsl + TWOL - r) - c;
            } else {
                return -(columnsl * (rowsl + TWOL - r) - c + TWOL);
            }
        } else if ((c == ONEL) || (c == columnsl + ONEL)) {
            return Long.MIN_VALUE;
        } else if (c == columnsl) {
            return ONEL;
        } else if (c < columnsl) {
            return c;
        } else if (cmod2 == ZEROL) {
            return (columnsl << ONEL) - c;
        } else {
            return -((columnsl << ONEL) - c + TWOL);
        }
    }

    /**
     * Sets the specified Fourier mode of the transformed data. The data array
     * results from a call to {@link DoubleFFT_2D#realForward(double[])}.
     *  
     * @param val
     *               the new value of the <code>[r][c]</code> Fourier mode
     * @param r
     *               the row index
     * @param c
     *               the column index
     * @param packed
     *               the transformed data
     * @param pos
     *               index of the first element in array <code>packed</code>
     */
    public void pack(final double val, final int r, final int c,
                     final double[] packed, final int pos)
    {
        final int index = getIndex(r, c);
        if (index >= 0) {
            packed[pos + index] = val;
        } else if (index > Integer.MIN_VALUE) {
            packed[pos - index] = -val;
        } else {
            throw new IllegalArgumentException(
                String.format(
                    "[%d][%d] component cannot be modified (always zero)",
                    r, c));
        }
    }

    /**
     * Sets the specified Fourier mode of the transformed data. The data array
     * results from a call to {@link DoubleFFT_2D#realForward(DoubleLargeArray)}.
     *  
     * @param val
     *               the new value of the <code>[r][c]</code> Fourier mode
     * @param r
     *               the row index
     * @param c
     *               the column index
     * @param packed
     *               the transformed data
     * @param pos
     *               index of the first element in array <code>packed</code>
     */
    public void pack(final double val, final long r, final long c,
                     final DoubleLargeArray packed, final long pos)
    {
        final long index = getIndex(r, c);
        if (index >= 0) {
            packed.setDouble(pos + index, val);
        } else if (index > Long.MIN_VALUE) {
            packed.setDouble(pos - index, -val);
        } else {
            throw new IllegalArgumentException(
                String.format(
                    "[%d][%d] component cannot be modified (always zero)",
                    r, c));
        }
    }

    /**
     * Sets the specified Fourier mode of the transformed data. The data array
     * results from a call to {@link DoubleFFT_2D#realForward(double[][])}.
     *  
     * @param val
     *               the new value of the <code>[r][c]</code> Fourier mode
     * @param r
     *               the row index
     * @param c
     *               the column index
     * @param packed
     *               the transformed data
     */
    public void pack(final double val, final int r, final int c,
                     final double[][] packed)
    {
        final int index = getIndex(r, c);
        if (index >= 0) {
            packed[index / columns][index % columns] = val;
        } else if (index > Integer.MIN_VALUE) {
            packed[(-index) / columns][(-index) % columns] = -val;
        } else {
            throw new IllegalArgumentException(
                String.format(
                    "[%d][%d] component cannot be modified (always zero)",
                    r, c));
        }
    }

    /**
     * Sets the specified Fourier mode of the transformed data. The data array
     * results from a call to {@link FloatFFT_2D#realForward(float[])}.
     *  
     * @param val
     *               the new value of the <code>[r][c]</code> Fourier mode
     * @param r
     *               the row index
     * @param c
     *               the column index
     * @param packed
     *               the transformed data
     * @param pos
     *               index of the first element in array <code>packed</code>
     */
    public void pack(final float val, final int r, final int c,
                     final float[] packed, final int pos)
    {
        final int index = getIndex(r, c);
        if (index >= 0) {
            packed[pos + index] = val;
        } else if (index > Integer.MIN_VALUE) {
            packed[pos - index] = -val;
        } else {
            throw new IllegalArgumentException(
                String.format(
                    "[%d][%d] component cannot be modified (always zero)",
                    r, c));
        }
    }

    /**
     * Sets the specified Fourier mode of the transformed data. The data array
     * results from a call to {@link FloatFFT_2D#realForward(FloatLargeArray)}.
     *  
     * @param val
     *               the new value of the <code>[r][c]</code> Fourier mode
     * @param r
     *               the row index
     * @param c
     *               the column index
     * @param packed
     *               the transformed data
     * @param pos
     *               index of the first element in array <code>packed</code>
     */
    public void pack(final float val, final long r, final long c,
                     final FloatLargeArray packed, final long pos)
    {
        final long index = getIndex(r, c);
        if (index >= 0) {
            packed.setFloat(pos + index, val);
        } else if (index > Long.MIN_VALUE) {
            packed.setFloat(pos - index, -val);
        } else {
            throw new IllegalArgumentException(
                String.format(
                    "[%d][%d] component cannot be modified (always zero)",
                    r, c));
        }
    }

    /**
     * Sets the specified Fourier mode of the transformed data. The data array
     * results from a call to {@link FloatFFT_2D#realForward(float[][])}.
     *  
     * @param val
     *               the new value of the <code>[r][c]</code> Fourier mode
     * @param r
     *               the row index
     * @param c
     *               the column index
     * @param packed
     *               the transformed data
     */
    public void pack(final float val, final int r, final int c,
                     final float[][] packed)
    {
        final int index = getIndex(r, c);
        if (index >= 0) {
            packed[index / columns][index % columns] = val;
        } else if (index > Integer.MIN_VALUE) {
            packed[(-index) / columns][(-index) % columns] = -val;
        } else {
            throw new IllegalArgumentException(
                String.format(
                    "[%d][%d] component cannot be modified (always zero)",
                    r, c));
        }
    }

    /**
     * Returns the specified Fourier mode of the transformed data. The data
     * array results from a call to {@link DoubleFFT_2D#realForward(double[])}.
     *  
     * @param r
     *               the row index
     * @param c
     *               the column index
     * @param packed
     *               the transformed data
     * @param pos
     *               index of the first element in array <code>packed</code>
     *  
     * @return the value of the <code>[r][c]</code> Fourier mode
     */
    public double unpack(final int r, final int c, final double[] packed,
                         final int pos)
    {
        final int index = getIndex(r, c);
        if (index >= 0) {
            return packed[pos + index];
        } else if (index > Integer.MIN_VALUE) {
            return -packed[pos - index];
        } else {
            return ZERO;
        }
    }

    /**
     * Returns the specified Fourier mode of the transformed data. The data
     * array results from a call to {@link DoubleFFT_2D#realForward(DoubleLargeArray)}.
     *  
     * @param r
     *               the row index
     * @param c
     *               the column index
     * @param packed
     *               the transformed data
     * @param pos
     *               index of the first element in array <code>packed</code>
     *  
     * @return the value of the <code>[r][c]</code> Fourier mode
     */
    public double unpack(final long r, final long c, final DoubleLargeArray packed,
                         final long pos)
    {
        final long index = getIndex(r, c);
        if (index >= 0) {
            return packed.getDouble(pos + index);
        } else if (index > Long.MIN_VALUE) {
            return -packed.getDouble(pos - index);
        } else {
            return ZEROL;
        }
    }

    /**
     * Returns the specified Fourier mode of the transformed data. The data
     * array results from a call to {@link DoubleFFT_2D#realForward(double[][])}
     * .
     *  
     * @param r
     *               the row index
     * @param c
     *               the column index
     * @param packed
     *               the transformed data
     *  
     * @return the value of the <code>[r][c]</code> Fourier mode
     */
    public double unpack(final int r, final int c, final double[][] packed)
    {
        final int index = getIndex(r, c);
        if (index >= 0) {
            return packed[index / columns][index % columns];
        } else if (index > Integer.MIN_VALUE) {
            return -packed[(-index) / columns][(-index) % columns];
        } else {
            return ZERO;
        }
    }

    /**
     * Returns the specified Fourier mode of the transformed data. The data
     * array results from a call to {@link FloatFFT_2D#realForward(float[])}
     * .
     *  
     * @param r
     *               the row index
     * @param c
     *               the column index
     * @param packed
     *               the transformed data
     * @param pos
     *               index of the first element in array <code>packed</code>
     *  
     * @return the value of the <code>[r][c]</code> Fourier mode
     */
    public float unpack(final int r, final int c, final float[] packed,
                        final int pos)
    {
        final int index = getIndex(r, c);
        if (index >= 0) {
            return packed[pos + index];
        } else if (index > Integer.MIN_VALUE) {
            return -packed[pos - index];
        } else {
            return ZERO;
        }
    }

    /**
     * Returns the specified Fourier mode of the transformed data. The data
     * array results from a call to {@link FloatFFT_2D#realForward(FloatLargeArray)}
     * .
     *  
     * @param r
     *               the row index
     * @param c
     *               the column index
     * @param packed
     *               the transformed data
     * @param pos
     *               index of the first element in array <code>packed</code>
     *  
     * @return the value of the <code>[r][c]</code> Fourier mode
     */
    public float unpack(final long r, final long c, final FloatLargeArray packed,
                        final long pos)
    {
        final long index = getIndex(r, c);
        if (index >= 0) {
            return packed.getFloat(pos + index);
        } else if (index > Long.MIN_VALUE) {
            return -packed.getFloat(pos - index);
        } else {
            return ZEROL;
        }
    }

    /**
     * Returns the specified Fourier mode of the transformed data. The data
     * array results from a call to {@link FloatFFT_2D#realForward(float[][])} .
     *  
     * @param r
     *               the row index
     * @param c
     *               the column index
     * @param packed
     *               the transformed data
     *  
     * @return the value of the <code>[r][c]</code> Fourier mode
     */
    public float unpack(final int r, final int c, final float[][] packed)
    {
        final int index = getIndex(r, c);
        if (index >= 0) {
            return packed[index / columns][index % columns];
        } else if (index > Integer.MIN_VALUE) {
            return -packed[(-index) / columns][(-index) % columns];
        } else {
            return ZERO;
        }
    }
}
