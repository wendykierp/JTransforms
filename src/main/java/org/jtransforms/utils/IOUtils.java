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
package org.jtransforms.utils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Date;
import java.util.Random;
import pl.edu.icm.jlargearrays.DoubleLargeArray;
import pl.edu.icm.jlargearrays.FloatLargeArray;
import static org.apache.commons.math3.util.FastMath.*;

/**
 * I/O utilities.
 *
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class IOUtils
{

    private static final String FF = "%.4f";

    private IOUtils()
    {

    }

    /**
     * Computes root mean square error between a and b.
     *
     * @param a input parameter
     * @param b input parameter
     *
     * @return root mean squared error between a and b
     */
    public static double computeRMSE(float a, float b)
    {
        double tmp = a - b;
        double rms = tmp * tmp;
        return sqrt(rms);
    }

    /**
     * Computes root mean square error between a and b.
     *
     * @param a input parameter
     * @param b input parameter
     *
     * @return root mean squared error between a and b
     */
    public static double computeRMSE(float[] a, float[] b)
    {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Arrays are not the same size");
        }
        double rms = 0;
        double tmp;
        for (int i = 0; i < a.length; i++) {
            tmp = (a[i] - b[i]);
            rms += tmp * tmp;
        }
        return sqrt(rms / a.length);
    }

    /**
     * Computes root mean square error between a and b.
     *
     * @param a input parameter
     * @param b input parameter
     *
     * @return root mean squared error between a and b
     */
    public static double computeRMSE(FloatLargeArray a, FloatLargeArray b)
    {
        if (a.length() != b.length()) {
            throw new IllegalArgumentException("Arrays are not the same size.");
        }
        double rms = 0;
        double tmp;
        for (long i = 0; i < a.length(); i++) {
            tmp = (a.getFloat(i) - b.getFloat(i));
            rms += tmp * tmp;
        }
        return sqrt(rms / (double) a.length());
    }

    /**
     * Computes root mean square error between a and b.
     *
     * @param a input parameter
     * @param b input parameter
     *
     * @return root mean squared error between a and b
     */
    public static double computeRMSE(float[][] a, float[][] b)
    {
        if (a.length != b.length || a[0].length != b[0].length) {
            throw new IllegalArgumentException("Arrays are not the same size");
        }
        double rms = 0;
        double tmp;
        for (int r = 0; r < a.length; r++) {
            for (int c = 0; c < a[0].length; c++) {
                tmp = (a[r][c] - b[r][c]);
                rms += tmp * tmp;
            }
        }
        return sqrt(rms / (a.length * a[0].length));
    }

    /**
     * Computes root mean square error between a and b.
     *
     * @param a input parameter
     * @param b input parameter
     *
     * @return root mean squared error between a and b
     */
    public static double computeRMSE(float[][][] a, float[][][] b)
    {
        if (a.length != b.length || a[0].length != b[0].length || a[0][0].length != b[0][0].length) {
            throw new IllegalArgumentException("Arrays are not the same size");
        }
        double rms = 0;
        double tmp;
        for (int s = 0; s < a.length; s++) {
            for (int r = 0; r < a[0].length; r++) {
                for (int c = 0; c < a[0][0].length; c++) {
                    tmp = (a[s][r][c] - b[s][r][c]);
                    rms += tmp * tmp;
                }
            }
        }
        return sqrt(rms / (a.length * a[0].length * a[0][0].length));
    }

    /**
     * Computes root mean square error between a and b.
     *
     * @param a input parameter
     * @param b input parameter
     *
     * @return root mean squared error between a and b
     */
    public static double computeRMSE(double a, double b)
    {
        double tmp = a - b;
        double rms = tmp * tmp;
        return sqrt(rms);
    }

    /**
     * Computes root mean square error between a and b.
     *
     * @param a input parameter
     * @param b input parameter
     *
     * @return root mean squared error between a and b
     */
    public static double computeRMSE(double[] a, double[] b)
    {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Arrays are not the same size");
        }
        double rms = 0;
        double tmp;
        for (int i = 0; i < a.length; i++) {
            tmp = (a[i] - b[i]);
            rms += tmp * tmp;
        }
        return sqrt(rms / a.length);
    }

    /**
     * Computes root mean square error between a and b.
     *
     * @param a input parameter
     * @param b input parameter
     *
     * @return root mean squared error between a and b
     */
    public static double computeRMSE(DoubleLargeArray a, DoubleLargeArray b)
    {
        if (a.length() != b.length()) {
            throw new IllegalArgumentException("Arrays are not the same size.");
        }
        double rms = 0;
        double tmp;
        for (long i = 0; i < a.length(); i++) {
            tmp = (a.getDouble(i) - b.getDouble(i));
            rms += tmp * tmp;
        }
        return sqrt(rms / (double) a.length());
    }

    /**
     * Computes root mean square error between a and b.
     *
     * @param a input parameter
     * @param b input parameter
     *
     * @return root mean squared error between a and b
     */
    public static double computeRMSE(double[][] a, double[][] b)
    {
        if (a.length != b.length || a[0].length != b[0].length) {
            throw new IllegalArgumentException("Arrays are not the same size");
        }
        double rms = 0;
        double tmp;
        for (int r = 0; r < a.length; r++) {
            for (int c = 0; c < a[0].length; c++) {
                tmp = (a[r][c] - b[r][c]);
                rms += tmp * tmp;
            }
        }
        return sqrt(rms / (a.length * a[0].length));
    }

    /**
     * Computes root mean square error between a and b.
     *
     * @param a input parameter
     * @param b input parameter
     *
     * @return root mean squared error between a and b
     */
    public static double computeRMSE(double[][][] a, double[][][] b)
    {
        if (a.length != b.length || a[0].length != b[0].length || a[0][0].length != b[0][0].length) {
            throw new IllegalArgumentException("Arrays are not the same size");
        }
        double rms = 0;
        double tmp;
        for (int s = 0; s < a.length; s++) {
            for (int r = 0; r < a[0].length; r++) {
                for (int c = 0; c < a[0][0].length; c++) {
                    tmp = (a[s][r][c] - b[s][r][c]);
                    rms += tmp * tmp;
                }
            }
        }
        return sqrt(rms / (a.length * a[0].length * a[0][0].length));
    }

    /**
     * Fills 1D matrix with random numbers.
     *
     * @param N size
     * @param m 1D matrix
     */
    public static void fillMatrix_1D(long N, double[] m)
    {
        Random r = new Random(2);
        for (int i = 0; i < N; i++) {
            m[i] = r.nextDouble();
        }
    }

    /**
     * Fills 1D matrix with random numbers.
     *
     * @param N size
     * @param m 1D matrix
     */
    public static void fillMatrix_1D(long N, DoubleLargeArray m)
    {
        Random r = new Random(2);
        for (long i = 0; i < N; i++) {
            m.setDouble(i, r.nextDouble());
        }
    }

    /**
     * Fills 1D matrix with random numbers.
     *
     * @param N size
     * @param m 1D matrix
     */
    public static void fillMatrix_1D(long N, FloatLargeArray m)
    {
        Random r = new Random(2);
        for (long i = 0; i < N; i++) {
            m.setDouble(i, r.nextFloat());
        }
    }

    /**
     * Fills 1D matrix with random numbers.
     *
     * @param N size
     * @param m 1D matrix
     */
    public static void fillMatrix_1D(long N, float[] m)
    {
        Random r = new Random(2);
        for (int i = 0; i < N; i++) {
            m[i] = r.nextFloat();
        }
    }

    /**
     * Fills 2D matrix with random numbers.
     *
     * @param n1 rows
     * @param n2 columns
     * @param m  2D matrix
     */
    public static void fillMatrix_2D(long n1, long n2, double[] m)
    {
        Random r = new Random(2);
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                m[(int) (i * n2 + j)] = r.nextDouble();
            }
        }
    }

    /**
     * Fills 2D matrix with random numbers.
     *
     * @param n1 rows
     * @param n2 columns
     * @param m  2D matrix
     */
    public static void fillMatrix_2D(long n1, long n2, FloatLargeArray m)
    {
        Random r = new Random(2);
        for (long i = 0; i < n1; i++) {
            for (long j = 0; j < n2; j++) {
                m.setFloat(i * n2 + j, r.nextFloat());
            }
        }
    }

    /**
     * Fills 2D matrix with random numbers.
     *
     * @param n1 rows
     * @param n2 columns
     * @param m  2D matrix
     */
    public static void fillMatrix_2D(long n1, long n2, DoubleLargeArray m)
    {
        Random r = new Random(2);
        for (long i = 0; i < n1; i++) {
            for (long j = 0; j < n2; j++) {
                m.setDouble(i * n2 + j, r.nextDouble());
            }
        }
    }

    /**
     * Fills 2D matrix with random numbers.
     *
     * @param n1 rows
     * @param n2 columns
     * @param m  2D matrix
     */
    public static void fillMatrix_2D(long n1, long n2, float[] m)
    {
        Random r = new Random(2);
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                m[(int) (i * n2 + j)] = r.nextFloat();
            }
        }
    }

    /**
     * Fills 2D matrix with random numbers.
     *
     * @param n1 rows
     * @param n2 columns
     * @param m  2D matrix
     */
    public static void fillMatrix_2D(long n1, long n2, double[][] m)
    {
        Random r = new Random(2);
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                m[i][j] = r.nextDouble();
            }
        }
    }

    /**
     * Fills 2D matrix with random numbers.
     *
     * @param n1 rows
     * @param n2 columns
     * @param m  2D matrix
     */
    public static void fillMatrix_2D(long n1, long n2, float[][] m)
    {
        Random r = new Random(2);
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                m[i][j] = r.nextFloat();
            }
        }
    }

    /**
     * Fills 3D matrix with random numbers.
     *
     * @param n1 slices
     * @param n2 rows
     * @param n3 columns
     * @param m  3D matrix
     */
    public static void fillMatrix_3D(long n1, long n2, long n3, double[] m)
    {
        Random r = new Random(2);
        long sliceStride = n2 * n3;
        long rowStride = n3;
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                for (int k = 0; k < n3; k++) {
                    m[(int) (i * sliceStride + j * rowStride + k)] = r.nextDouble();
                }
            }
        }
    }

    /**
     * Fills 3D matrix with random numbers.
     *
     * @param n1 slices
     * @param n2 rows
     * @param n3 columns
     * @param m  3D matrix
     */
    public static void fillMatrix_3D(long n1, long n2, long n3, DoubleLargeArray m)
    {
        Random r = new Random(2);
        long sliceStride = n2 * n3;
        long rowStride = n3;
        for (long i = 0; i < n1; i++) {
            for (long j = 0; j < n2; j++) {
                for (long k = 0; k < n3; k++) {
                    m.setDouble(i * sliceStride + j * rowStride + k, r.nextDouble());
                }
            }
        }
    }

    /**
     * Fills 3D matrix with random numbers.
     *
     * @param n1 slices
     * @param n2 rows
     * @param n3 columns
     * @param m  3D matrix
     */
    public static void fillMatrix_3D(long n1, long n2, long n3, FloatLargeArray m)
    {
        Random r = new Random(2);
        long sliceStride = n2 * n3;
        long rowStride = n3;
        for (long i = 0; i < n1; i++) {
            for (long j = 0; j < n2; j++) {
                for (long k = 0; k < n3; k++) {
                    m.setDouble(i * sliceStride + j * rowStride + k, r.nextFloat());
                }
            }
        }
    }

    /**
     * Fills 3D matrix with random numbers.
     *
     * @param n1 slices
     * @param n2 rows
     * @param n3 columns
     * @param m  3D matrix
     */
    public static void fillMatrix_3D(long n1, long n2, long n3, float[] m)
    {
        Random r = new Random(2);
        long sliceStride = n2 * n3;
        long rowStride = n3;
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                for (int k = 0; k < n3; k++) {
                    m[(int) (i * sliceStride + j * rowStride + k)] = r.nextFloat();
                }
            }
        }
    }

    /**
     * Fills 3D matrix with random numbers.
     *
     * @param n1 slices
     * @param n2 rows
     * @param n3 columns
     * @param m  3D matrix
     */
    public static void fillMatrix_3D(long n1, long n2, long n3, double[][][] m)
    {
        Random r = new Random(2);
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                for (int k = 0; k < n3; k++) {
                    m[i][j][k] = r.nextDouble();
                }
            }
        }
    }

    /**
     * Fills 3D matrix with random numbers.
     *
     * @param n1 slices
     * @param n2 rows
     * @param n3 columns
     * @param m  3D matrix
     */
    public static void fillMatrix_3D(long n1, long n2, long n3, float[][][] m)
    {
        Random r = new Random(2);
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                for (int k = 0; k < n3; k++) {
                    m[i][j][k] = r.nextFloat();
                }
            }
        }
    }

    /**
     * Displays elements of <code>x</code>, assuming that it is 1D complex
     * array. Complex data is represented by 2 double values in sequence: the
     * real and imaginary parts.
     *
     * @param x     input array
     * @param title title of the array
     */
    public static void showComplex_1D(double[] x, String title)
    {
        System.out.println(title);
        System.out.println("-------------------");
        for (int i = 0; i < x.length; i = i + 2) {
            if (x[i + 1] == 0) {
                System.out.println(String.format(FF, x[i]));
                continue;
            }
            if (x[i] == 0) {
                System.out.println(String.format(FF, x[i + 1]) + "i");
                continue;
            }
            if (x[i + 1] < 0) {
                System.out.println(String.format(FF, x[i]) + " - " + (String.format(FF, -x[i + 1])) + "i");
                continue;
            }
            System.out.println(String.format(FF, x[i]) + " + " + (String.format(FF, x[i + 1])) + "i");
        }
        System.out.println();
    }

    /**
     * Displays elements of <code>x</code>, assuming that it is 2D complex
     * array. Complex data is represented by 2 double values in sequence: the
     * real and imaginary parts.
     *
     * @param rows    number of rows in the input array
     * @param columns number of columns in the input array
     * @param x       input array
     * @param title   title of the array
     */
    public static void showComplex_2D(int rows, int columns, double[] x, String title)
    {
        StringBuilder s = new StringBuilder(String.format(title + ": complex array 2D: %d rows, %d columns\n\n", rows, columns));
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < 2 * columns; c = c + 2) {
                if (x[r * 2 * columns + c + 1] == 0) {
                    s.append(String.format(FF + "\t", x[r * 2 * columns + c]));
                    continue;
                }
                if (x[r * 2 * columns + c] == 0) {
                    s.append(String.format(FF + "i\t", x[r * 2 * columns + c + 1]));
                    continue;
                }
                if (x[r * 2 * columns + c + 1] < 0) {
                    s.append(String.format(FF + " - " + FF + "i\t", x[r * 2 * columns + c], -x[r * 2 * columns + c + 1]));
                    continue;
                }
                s.append(String.format(FF + " + " + FF + "i\t", x[r * 2 * columns + c], x[r * 2 * columns + c + 1]));
            }
            s.append("\n");
        }
        System.out.println(s.toString());
    }

    /**
     * Displays elements of <code>x</code>, assuming that it is 2D complex
     * array. Complex data is represented by 2 double values in sequence: the
     * real and imaginary parts.
     *
     * @param x     input array
     * @param title title of the array
     */
    public static void showComplex_2D(double[][] x, String title)
    {
        int rows = x.length;
        int columns = x[0].length;
        StringBuilder s = new StringBuilder(String.format(title + ": complex array 2D: %d rows, %d columns\n\n", rows, columns));
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c = c + 2) {
                if (x[r][c + 1] == 0) {
                    s.append(String.format(FF + "\t", x[r][c]));
                    continue;
                }
                if (x[r][c] == 0) {
                    s.append(String.format(FF + "i\t", x[r][c + 1]));
                    continue;
                }
                if (x[r][c + 1] < 0) {
                    s.append(String.format(FF + " - " + FF + "i\t", x[r][c], -x[r][c + 1]));
                    continue;
                }
                s.append(String.format(FF + " + " + FF + "i\t", x[r][c], x[r][c + 1]));
            }
            s.append("\n");
        }
        System.out.println(s.toString());
    }

    /**
     * Displays elements of <code>x</code>, assuming that it is 3D complex
     * array. Complex data is represented by 2 double values in sequence: the
     * real and imaginary parts.
     *
     * @param n1    first dimension
     * @param n2    second dimension
     * @param n3    third dimension
     * @param x     input array
     * @param title title of the array
     */
    public static void showComplex_3D(int n1, int n2, int n3, double[] x, String title)
    {
        int sliceStride = n2 * 2 * n3;
        int rowStride = 2 * n3;

        System.out.println(title);
        System.out.println("-------------------");

        for (int k = 0; k < 2 * n3; k = k + 2) {
            System.out.println("(:,:," + k / 2 + ")=\n");
            for (int i = 0; i < n1; i++) {
                for (int j = 0; j < n2; j++) {
                    if (x[i * sliceStride + j * rowStride + k + 1] == 0) {
                        System.out.print(String.format(FF, x[i * sliceStride + j * rowStride + k]) + "\t");
                        continue;
                    }
                    if (x[i * sliceStride + j * rowStride + k] == 0) {
                        System.out.print(String.format(FF, x[i * sliceStride + j * rowStride + k + 1]) + "i\t");
                        continue;
                    }
                    if (x[i * sliceStride + j * rowStride + k + 1] < 0) {
                        System.out.print(String.format(FF, x[i * sliceStride + j * rowStride + k]) + " - " + String.format(FF, -x[i * sliceStride + j * rowStride + k + 1]) + "i\t");
                        continue;
                    }
                    System.out.print(String.format(FF, x[i * sliceStride + j * rowStride + k]) + " + " + String.format(FF, x[i * sliceStride + j * rowStride + k + 1]) + "i\t");
                }
                System.out.println("");
            }
        }
        System.out.println("");
    }

    /**
     * Displays elements of <code>x</code>. Complex data is represented by 2
     * double values in sequence: the real and imaginary parts.
     *
     * @param x     input array
     * @param title title of the array
     */
    public static void showComplex_3D(double[][][] x, String title)
    {
        System.out.println(title);
        System.out.println("-------------------");
        int slices = x.length;
        int rows = x[0].length;
        int columns = x[0][0].length;
        for (int k = 0; k < columns; k = k + 2) {
            System.out.println("(:,:," + k / 2 + ")=\n");
            for (int i = 0; i < slices; i++) {
                for (int j = 0; j < rows; j++) {
                    if (x[i][j][k + 1] == 0) {
                        System.out.print(String.format(FF, x[i][j][k]) + "\t");
                        continue;
                    }
                    if (x[i][j][k] == 0) {
                        System.out.print(String.format(FF, x[i][j][k + 1]) + "i\t");
                        continue;
                    }
                    if (x[i][j][k + 1] < 0) {
                        System.out.print(String.format(FF, x[i][j][k]) + " - " + String.format(FF, -x[i][j][k + 1]) + "i\t");
                        continue;
                    }
                    System.out.print(String.format(FF, x[i][j][k]) + " + " + String.format(FF, x[i][j][k + 1]) + "i\t");
                }
                System.out.println("");
            }
        }
        System.out.println("");
    }

    /**
     * Displays elements of <code>x</code>, assuming that it is 3D complex
     * array. Complex data is represented by 2 double values in sequence: the
     * real and imaginary parts.
     *
     * @param n1    first dimension
     * @param n2    second dimension
     * @param n3    third dimension
     * @param x     input array
     * @param title title of the array
     */
    public static void showComplex_3D(int n1, int n2, int n3, float[] x, String title)
    {
        int sliceStride = n2 * 2 * n3;
        int rowStride = 2 * n3;

        System.out.println(title);
        System.out.println("-------------------");

        for (int k = 0; k < 2 * n3; k = k + 2) {
            System.out.println("(:,:," + k / 2 + ")=\n");
            for (int i = 0; i < n1; i++) {
                for (int j = 0; j < n2; j++) {
                    if (x[i * sliceStride + j * rowStride + k + 1] == 0) {
                        System.out.print(String.format(FF, x[i * sliceStride + j * rowStride + k]) + "\t");
                        continue;
                    }
                    if (x[i * sliceStride + j * rowStride + k] == 0) {
                        System.out.print(String.format(FF, x[i * sliceStride + j * rowStride + k + 1]) + "i\t");
                        continue;
                    }
                    if (x[i * sliceStride + j * rowStride + k + 1] < 0) {
                        System.out.print(String.format(FF, x[i * sliceStride + j * rowStride + k]) + " - " + String.format(FF, -x[i * sliceStride + j * rowStride + k + 1]) + "i\t");
                        continue;
                    }
                    System.out.print(String.format(FF, x[i * sliceStride + j * rowStride + k]) + " + " + String.format(FF, x[i * sliceStride + j * rowStride + k + 1]) + "i\t");
                }
                System.out.println("");
            }
        }
        System.out.println("");
    }

    /**
     * Displays elements of <code>x</code>, assuming that it is 1D real array.
     *
     * @param x     input array
     * @param title title of the array
     */
    public static void showReal_1D(double[] x, String title)
    {
        System.out.println(title);
        System.out.println("-------------------");
        for (int j = 0; j < x.length; j++) {
            System.out.println(String.format(FF, x[j]));
        }
        System.out.println();
    }

    /**
     * Displays elements of <code>x</code>, assuming that it is 2D real array.
     *
     * @param n1    first dimension
     * @param n2    second dimension
     * @param x     input array
     * @param title title of the array
     *
     */
    public static void showReal_2D(int n1, int n2, double[] x, String title)
    {
        System.out.println(title);
        System.out.println("-------------------");
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                if (abs(x[i * n2 + j]) < 5e-5) {
                    System.out.print("0\t");
                } else {
                    System.out.print(String.format(FF, x[i * n2 + j]) + "\t");
                }
            }
            System.out.println();
        }
        System.out.println();
    }

    /**
     * Displays elements of <code>x</code>, assuming that it is 3D real array.
     *
     *
     * @param n1    first dimension
     * @param n2    second dimension
     * @param n3    third dimension
     * @param x     input array
     * @param title title of the array
     */
    public static void showReal_3D(int n1, int n2, int n3, double[] x, String title)
    {
        int sliceStride = n2 * n3;
        int rowStride = n3;

        System.out.println(title);
        System.out.println("-------------------");

        for (int k = 0; k < n3; k++) {
            System.out.println();
            System.out.println("(:,:," + k + ")=\n");
            for (int i = 0; i < n1; i++) {
                for (int j = 0; j < n2; j++) {
                    if (abs(x[i * sliceStride + j * rowStride + k]) <= 5e-5) {
                        System.out.print("0\t");
                    } else {
                        System.out.print(String.format(FF, x[i * sliceStride + j * rowStride + k]) + "\t");
                    }
                }
                System.out.println();
            }
        }
        System.out.println();
    }

    /**
     * Displays elements of <code>x</code>.
     *
     * @param x     input array
     * @param title title of the array
     */
    public static void showReal_3D(double[][][] x, String title)
    {

        System.out.println(title);
        System.out.println("-------------------");
        int slices = x.length;
        int rows = x[0].length;
        int columns = x[0][0].length;
        for (int k = 0; k < columns; k++) {
            System.out.println();
            System.out.println("(:,:," + k + ")=\n");
            for (int i = 0; i < slices; i++) {
                for (int j = 0; j < rows; j++) {
                    if (abs(x[i][j][k]) <= 5e-5) {
                        System.out.print("0\t");
                    } else {
                        System.out.print(String.format(FF, x[i][j][k]) + "\t");
                    }
                }
                System.out.println();
            }
        }
        System.out.println();
    }

    /**
     * Saves elements of <code>x</code> in a file <code>filename</code>,
     * assuming that it is 1D complex array. Complex data is represented by 2
     * double values in sequence: the real and imaginary parts.
     *
     * @param x        input array
     * @param filename finename
     */
    public static void writeToFileComplex_1D(double[] x, String filename)
    {
        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(filename));
            for (int i = 0; i < x.length; i = i + 2) {
                if (x[i + 1] == 0) {
                    out.write(String.format(FF, x[i]));
                    out.newLine();
                    continue;
                }
                if (x[i] == 0) {
                    out.write(String.format(FF, x[i + 1]) + "i");
                    out.newLine();
                    continue;
                }
                if (x[i + 1] < 0) {
                    out.write(String.format(FF, x[i]) + " - " + String.format(FF, -x[i + 1]) + "i");
                    out.newLine();
                    continue;
                }
                out.write(String.format(FF, x[i]) + " + " + String.format(FF, x[i + 1]) + "i");
                out.newLine();
            }
            out.newLine();
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Saves elements of <code>x</code> in a file <code>filename</code>,
     * assuming that it is 1D complex array. Complex data is represented by 2
     * double values in sequence: the real and imaginary parts.
     *
     * @param x        input array
     * @param filename finename
     */
    public static void writeToFileComplex_1D(float[] x, String filename)
    {
        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(filename));
            for (int i = 0; i < x.length; i = i + 2) {
                if (x[i + 1] == 0) {
                    out.write(String.format(FF, x[i]));
                    out.newLine();
                    continue;
                }
                if (x[i] == 0) {
                    out.write(String.format(FF, x[i + 1]) + "i");
                    out.newLine();
                    continue;
                }
                if (x[i + 1] < 0) {
                    out.write(String.format(FF, x[i]) + " - " + String.format(FF, -x[i + 1]) + "i");
                    out.newLine();
                    continue;
                }
                out.write(String.format(FF, x[i]) + " + " + String.format(FF, x[i + 1]) + "i");
                out.newLine();
            }
            out.newLine();
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Saves elements of <code>x</code> in a file <code>filename</code>,
     * assuming that it is 2D complex array. Complex data is represented by 2
     * double values in sequence: the real and imaginary parts.
     *
     * @param n1       first dimension
     * @param n2       second dimension
     * @param x        input array
     * @param filename finename
     */
    public static void writeToFileComplex_2D(int n1, int n2, double[] x, String filename)
    {
        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(filename));
            for (int i = 0; i < n1; i++) {
                for (int j = 0; j < 2 * n2; j = j + 2) {
                    if ((abs(x[i * 2 * n2 + j]) < 5e-5) && (abs(x[i * 2 * n2 + j + 1]) < 5e-5)) {
                        if (x[i * 2 * n2 + j + 1] >= 0.0) {
                            out.write("0 + 0i\t");
                        } else {
                            out.write("0 - 0i\t");
                        }
                        continue;
                    }

                    if (abs(x[i * 2 * n2 + j + 1]) < 5e-5) {
                        if (x[i * 2 * n2 + j + 1] >= 0.0) {
                            out.write(String.format(FF, x[i * 2 * n2 + j]) + " + 0i\t");
                        } else {
                            out.write(String.format(FF, x[i * 2 * n2 + j]) + " - 0i\t");
                        }
                        continue;
                    }
                    if (abs(x[i * 2 * n2 + j]) < 5e-5) {
                        if (x[i * 2 * n2 + j + 1] >= 0.0) {
                            out.write("0 + " + String.format(FF, x[i * 2 * n2 + j + 1]) + "i\t");
                        } else {
                            out.write("0 - " + String.format(FF, -x[i * 2 * n2 + j + 1]) + "i\t");
                        }
                        continue;
                    }
                    if (x[i * 2 * n2 + j + 1] < 0) {
                        out.write(String.format(FF, x[i * 2 * n2 + j]) + " - " + String.format(FF, -x[i * 2 * n2 + j + 1]) + "i\t");
                        continue;
                    }
                    out.write(String.format(FF, x[i * 2 * n2 + j]) + " + " + String.format(FF, x[i * 2 * n2 + j + 1]) + "i\t");
                }
                out.newLine();
            }

            out.newLine();
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Saves elements of <code>x</code> in a file <code>filename</code>,
     * assuming that it is 2D complex array. Complex data is represented by 2
     * double values in sequence: the real and imaginary parts.
     *
     * @param n1       first dimension
     * @param n2       second dimension
     * @param x        input array
     * @param filename finename
     */
    public static void writeToFileComplex_2D(int n1, int n2, float[] x, String filename)
    {
        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(filename));
            for (int i = 0; i < n1; i++) {
                for (int j = 0; j < 2 * n2; j = j + 2) {
                    if ((abs(x[i * 2 * n2 + j]) < 5e-5) && (abs(x[i * 2 * n2 + j + 1]) < 5e-5)) {
                        if (x[i * 2 * n2 + j + 1] >= 0.0) {
                            out.write("0 + 0i\t");
                        } else {
                            out.write("0 - 0i\t");
                        }
                        continue;
                    }

                    if (abs(x[i * 2 * n2 + j + 1]) < 5e-5) {
                        if (x[i * 2 * n2 + j + 1] >= 0.0) {
                            out.write(String.format(FF, x[i * 2 * n2 + j]) + " + 0i\t");
                        } else {
                            out.write(String.format(FF, x[i * 2 * n2 + j]) + " - 0i\t");
                        }
                        continue;
                    }
                    if (abs(x[i * 2 * n2 + j]) < 5e-5) {
                        if (x[i * 2 * n2 + j + 1] >= 0.0) {
                            out.write("0 + " + String.format(FF, x[i * 2 * n2 + j + 1]) + "i\t");
                        } else {
                            out.write("0 - " + String.format(FF, -x[i * 2 * n2 + j + 1]) + "i\t");
                        }
                        continue;
                    }
                    if (x[i * 2 * n2 + j + 1] < 0) {
                        out.write(String.format(FF, x[i * 2 * n2 + j]) + " - " + String.format(FF, -x[i * 2 * n2 + j + 1]) + "i\t");
                        continue;
                    }
                    out.write(String.format(FF, x[i * 2 * n2 + j]) + " + " + String.format(FF, x[i * 2 * n2 + j + 1]) + "i\t");
                }
                out.newLine();
            }

            out.newLine();
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Saves elements of <code>x</code> in a file <code>filename</code>. Complex
     * data is represented by 2 double values in sequence: the real and
     * imaginary parts.
     *
     * @param x        input array
     * @param filename finename
     */
    public static void writeToFileComplex_2D(double[][] x, String filename)
    {
        int n1 = x.length;
        int n2 = x[0].length;
        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(filename));
            for (int i = 0; i < n1; i++) {
                for (int j = 0; j < 2 * n2; j = j + 2) {
                    if ((abs(x[i][j]) < 5e-5) && (abs(x[i][j + 1]) < 5e-5)) {
                        if (x[i][j + 1] >= 0.0) {
                            out.write("0 + 0i\t");
                        } else {
                            out.write("0 - 0i\t");
                        }
                        continue;
                    }

                    if (abs(x[i][j + 1]) < 5e-5) {
                        if (x[i][j + 1] >= 0.0) {
                            out.write(String.format(FF, x[i][j]) + " + 0i\t");
                        } else {
                            out.write(String.format(FF, x[i][j]) + " - 0i\t");
                        }
                        continue;
                    }
                    if (abs(x[i][j]) < 5e-5) {
                        if (x[i][j + 1] >= 0.0) {
                            out.write("0 + " + String.format(FF, x[i][j + 1]) + "i\t");
                        } else {
                            out.write("0 - " + String.format(FF, -x[i][j + 1]) + "i\t");
                        }
                        continue;
                    }
                    if (x[i][j + 1] < 0) {
                        out.write(String.format(FF, x[i][j]) + " - " + String.format(FF, -x[i][j + 1]) + "i\t");
                        continue;
                    }
                    out.write(String.format(FF, x[i][j]) + " + " + String.format(FF, x[i][j + 1]) + "i\t");
                }
                out.newLine();
            }

            out.newLine();
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Saves elements of <code>x</code> in a file <code>filename</code>,
     * assuming that it is 3D complex array. Complex data is represented by 2
     * double values in sequence: the real and imaginary parts.
     *
     * @param n1       first dimension
     * @param n2       second dimension
     * @param n3       third dimension
     * @param x        input array
     * @param filename finename
     */
    public static void writeToFileComplex_3D(int n1, int n2, int n3, double[] x, String filename)
    {
        int sliceStride = n2 * n3 * 2;
        int rowStride = n3 * 2;
        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(filename));
            for (int k = 0; k < 2 * n3; k = k + 2) {
                out.newLine();
                out.write("(:,:," + k / 2 + ")=");
                out.newLine();
                out.newLine();
                for (int i = 0; i < n1; i++) {
                    for (int j = 0; j < n2; j++) {
                        if (x[i * sliceStride + j * rowStride + k + 1] == 0) {
                            out.write(String.format(FF, x[i * sliceStride + j * rowStride + k]) + "\t");
                            continue;
                        }
                        if (x[i * sliceStride + j * rowStride + k] == 0) {
                            out.write(String.format(FF, x[i * sliceStride + j * rowStride + k + 1]) + "i\t");
                            continue;
                        }
                        if (x[i * sliceStride + j * rowStride + k + 1] < 0) {
                            out.write(String.format(FF, x[i * sliceStride + j * rowStride + k]) + " - " + String.format(FF, -x[i * sliceStride + j * rowStride + k + 1]) + "i\t");
                            continue;
                        }
                        out.write(String.format(FF, x[i * sliceStride + j * rowStride + k]) + " + " + String.format(FF, x[i * sliceStride + j * rowStride + k + 1]) + "i\t");
                    }
                    out.newLine();
                }
            }
            out.newLine();
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Saves elements of <code>x</code> in a file <code>filename</code>. Complex
     * data is represented by 2 double values in sequence: the real and
     * imaginary parts.
     *
     * @param x        input array
     * @param filename finename
     */
    public static void writeToFileComplex_3D(double[][][] x, String filename)
    {
        int n1 = x.length;
        int n2 = x[0].length;
        int n3 = x[0][0].length;
        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(filename));
            for (int k = 0; k < 2 * n3; k = k + 2) {
                out.newLine();
                out.write("(:,:," + k / 2 + ")=");
                out.newLine();
                out.newLine();
                for (int i = 0; i < n1; i++) {
                    for (int j = 0; j < n2; j++) {
                        if (x[i][j][k + 1] == 0) {
                            out.write(String.format(FF, x[i][j][k]) + "\t");
                            continue;
                        }
                        if (x[i][j][k] == 0) {
                            out.write(String.format(FF, x[i][j][k + 1]) + "i\t");
                            continue;
                        }
                        if (x[i][j][k + 1] < 0) {
                            out.write(String.format(FF, x[i][j][k]) + " - " + String.format(FF, -x[i][j][k + 1]) + "i\t");
                            continue;
                        }
                        out.write(String.format(FF, x[i][j][k]) + " + " + String.format(FF, x[i][j][k + 1]) + "i\t");
                    }
                    out.newLine();
                }
            }
            out.newLine();
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Saves elements of <code>x</code> in a file <code>filename</code>,
     * assuming that it is 2D real array.
     *
     * @param x        input array
     * @param filename finename
     */
    public static void writeToFileReal_1D(double[] x, String filename)
    {
        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(filename));
            for (int j = 0; j < x.length; j++) {
                out.write(String.format(FF, x[j]));
                out.newLine();
            }
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Saves elements of <code>x</code> in a file <code>filename</code>,
     * assuming that it is 2D real array.
     *
     * @param x        input array
     * @param filename finename
     */
    public static void writeToFileReal_1D(float[] x, String filename)
    {
        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(filename));
            for (int j = 0; j < x.length; j++) {
                out.write(String.format(FF, x[j]));
                out.newLine();
            }
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Saves elements of <code>x</code> in a file <code>filename</code>,
     * assuming that it is 2D real array.
     *
     * @param n1       first dimension
     * @param n2       second dimension
     * @param x        input array
     * @param filename finename
     */
    public static void writeToFileReal_2D(int n1, int n2, double[] x, String filename)
    {
        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(filename));
            for (int i = 0; i < n1; i++) {
                for (int j = 0; j < n2; j++) {
                    if (abs(x[i * n2 + j]) < 5e-5) {
                        out.write("0\t");
                    } else {
                        out.write(String.format(FF, x[i * n2 + j]) + "\t");
                    }
                }
                out.newLine();
            }
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Saves elements of <code>x</code> in a file <code>filename</code>,
     * assuming that it is 2D real array.
     *
     * @param n1       first dimension
     * @param n2       second dimension
     * @param x        input array
     * @param filename finename
     */
    public static void writeToFileReal_2D(int n1, int n2, float[] x, String filename)
    {
        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(filename));
            for (int i = 0; i < n1; i++) {
                for (int j = 0; j < n2; j++) {
                    if (abs(x[i * n2 + j]) < 5e-5) {
                        out.write("0\t");
                    } else {
                        out.write(String.format(FF, x[i * n2 + j]) + "\t");
                    }
                }
                out.newLine();
            }
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Saves elements of <code>x</code> in a file <code>filename</code>,
     * assuming that it is 3D real array.
     *
     * @param n1       first dimension
     * @param n2       second dimension
     * @param n3       third dimension
     * @param x        input array
     * @param filename finename
     */
    public static void writeToFileReal_3D(int n1, int n2, int n3, double[] x, String filename)
    {
        int sliceStride = n2 * n3;
        int rowStride = n3;

        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(filename));
            for (int k = 0; k < n3; k++) {
                out.newLine();
                out.write("(:,:," + k + ")=");
                out.newLine();
                out.newLine();
                for (int i = 0; i < n1; i++) {
                    for (int j = 0; j < n2; j++) {
                        out.write(String.format(FF, x[i * sliceStride + j * rowStride + k]) + "\t");
                    }
                    out.newLine();
                }
                out.newLine();
            }
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Saves benchmark results in a file.
     *
     * @param filename                  filename
     * @param nthread                   number of threads
     * @param niter                     number of iterations
     * @param doWarmup                  if warmup was performed
     * @param doScaling                 if scaling was performed
     * @param sizes                     benchmarked sizes
     * @param times_without_constructor timings excluding constructor
     * @param times_with_constructor    timings including constructor
     */
    public static void writeFFTBenchmarkResultsToFile(String filename, int nthread, int niter, boolean doWarmup, boolean doScaling, long[] sizes, double[] times_without_constructor, double[] times_with_constructor)
    {
        String[] properties = {"os.name", "os.version", "os.arch", "java.vendor", "java.version"};
        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(filename, false));
            out.write(new Date().toString());
            out.newLine();
            out.write("System properties:");
            out.newLine();
            out.write("\tos.name = " + System.getProperty(properties[0]));
            out.newLine();
            out.write("\tos.version = " + System.getProperty(properties[1]));
            out.newLine();
            out.write("\tos.arch = " + System.getProperty(properties[2]));
            out.newLine();
            out.write("\tjava.vendor = " + System.getProperty(properties[3]));
            out.newLine();
            out.write("\tjava.version = " + System.getProperty(properties[4]));
            out.newLine();
            out.write("\tavailable processors = " + Runtime.getRuntime().availableProcessors());
            out.newLine();
            out.write("Settings:");
            out.newLine();
            out.write("\tused processors = " + nthread);
            out.newLine();
            out.write("\tTHREADS_BEGIN_N_2D = " + CommonUtils.getThreadsBeginN_2D());
            out.newLine();
            out.write("\tTHREADS_BEGIN_N_3D = " + CommonUtils.getThreadsBeginN_3D());
            out.newLine();
            out.write("\tnumber of iterations = " + niter);
            out.newLine();
            out.write("\twarm-up performed = " + doWarmup);
            out.newLine();
            out.write("\tscaling performed = " + doScaling);
            out.newLine();
            out.write("--------------------------------------------------------------------------------------------------");
            out.newLine();
            out.write("sizes=[");
            for (int i = 0; i < sizes.length; i++) {
                out.write(Long.toString(sizes[i]));
                if (i < sizes.length - 1) {
                    out.write(", ");
                } else {
                    out.write("]");
                }
            }
            out.newLine();
            out.write("times without constructor(in msec)=[");
            for (int i = 0; i < times_without_constructor.length; i++) {
                out.write(String.format("%.2f", times_without_constructor[i]));
                if (i < times_without_constructor.length - 1) {
                    out.write(", ");
                } else {
                    out.write("]");
                }
            }
            out.newLine();
            out.write("times with constructor(in msec)=[");
            for (int i = 0; i < times_without_constructor.length; i++) {
                out.write(String.format("%.2f", times_with_constructor[i]));
                if (i < times_with_constructor.length - 1) {
                    out.write(", ");
                } else {
                    out.write("]");
                }
            }
            out.newLine();
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
