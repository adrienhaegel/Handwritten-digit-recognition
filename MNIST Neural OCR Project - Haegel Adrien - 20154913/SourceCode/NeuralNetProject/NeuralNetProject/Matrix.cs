using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OCRMNISTReader
{
    class Matrix
    {
        //A Matrix implementation
        public double[][] data;

        int nbrow;
        int nbcol;


        public Matrix(int nbrow, int nbcol)
        {
            this.nbrow = nbrow;
            this.nbcol = nbcol;
            data = new double[nbrow][];
            for (int k = 0; k < nbrow; k++)
            {
                data[k] = new double[nbcol];
            }

            for (int i = 0; i < nbrow; i++)
            {
                for (int j = 0; j < nbcol; j++)
                {
                    data[i][j] = 0;
                }
            }
        }

        public static Matrix Random(int nbrow, int nbcol, double deviation)
        {
            Matrix m = new Matrix(nbrow, nbcol);

            Random rand = new Random(); //reuse this if you are generating many


            for (int i = 0; i < nbrow; i++)
            {
                for (int j = 0; j < nbcol; j++)
                {
                    double u1 = rand.NextDouble(); //these are uniform(0,1) random doubles
                    double u2 = rand.NextDouble();
                    double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                                 Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
                    double randNormal =
                                 0 + deviation * randStdNormal; //random normal(mean,stdDev^2)


                    m.data[i][j] = randNormal;
                }
            }

            return m;
        }

        public void Clear()
        {
            for (int i = 0; i < nbrow; i++)
            {
                for (int j = 0; j < nbcol; j++)
                {
                    data[i][j] = 0;
                }
            }
        }


        public double Get(int row, int col)
        {
            return this.data[row][col];
        }

        public Matrix Transpose()
        {
            Matrix newmatrix = new Matrix(this.nbcol, this.nbrow);
            for (int i = 0; i < nbrow; i++)
            {
                for (int j = 0; j < nbcol; j++)
                {
                    newmatrix.data[j][i] = data[i][j];
                }
            }
            return newmatrix;
        }

        public double GetValue(int i, int j)
        {
            return this.data[i][j];
        }

        public Vector GetRow(int rowindex)
        {
            Vector v = new Vector(this.nbcol);
            v.data = this.data[rowindex];

            return v;
        }

        public Vector GetColumn(int colindex)
        {
            Vector v = new Vector(this.nbrow);
            for (int i = 0; i < this.nbrow; i++)
            {
                v.data[i] = this.data[i][colindex];
            }


            return v;
        }

        public void UpdateBackProp(Vector VL, Vector VR)
        {
            Parallel.For(0, VL.size, new ParallelOptions { MaxDegreeOfParallelism = 16 }, i =>
            {
                for (int j = 0; j < VR.size; j++)
                {
                    this.data[i][j] += VL.data[i] * VR.data[j];
                }
            });
            /*
            for (int i = 0; i < VL.size; i++)
            {
                for(int j = 0; j < VR.size; j++)
                {
                    this.data[i][j] += VL.data[i] * VR.data[j];
                }
                
            }
            */

        }

        public static Vector Forward_neuronz(Matrix weights, Vector activations, Vector bias)
        {
            Vector Vresult = new Vector(bias.size);
            //Vresult.Clear();
            Parallel.For(0, weights.nbrow,  new ParallelOptions { MaxDegreeOfParallelism = 16 }, i =>
            {
                double value = 0;
                for (int j = 0; j < activations.size; j++)
                {
                    value += weights.data[i][j] * activations.data[j];
                }

                Vresult.data[i] += value + bias.data[i];
 
            });
            return Vresult;

        }

        //NOT STATIC METHODS

        public void ScalarMultiply(double scalar)
        {
            Parallel.For(0, nbrow, new ParallelOptions { MaxDegreeOfParallelism = 16 }, i =>
            {

                for (int j = 0; j < nbcol; j++)
                {
                    data[i][j] *= scalar;
                }
            });
        }

        public void ScalarAddition(double scalar)
        {
            for (int i = 0; i < nbrow; i++)
            {
                for (int j = 0; j < nbcol; j++)
                {
                    data[i][j] += scalar;
                }
            }
        }


        public void TermbyTermAdd(Matrix Madd)
        {
            for (int i = 0; i < nbrow; i++)
            {
                for (int j = 0; j < nbcol; j++)
                {
                    data[i][j] += Madd.data[i][j];
                }
            }
        }



        //STATIC METHODS
        public static Matrix Multiply(Matrix ML, Matrix MR)
        {
            Matrix Mresult = new Matrix(ML.nbrow, MR.nbcol);

            for (int i = 0; i < ML.nbrow; i++)
            {
                for (int k = 0; k < ML.nbcol; k++)
                {
                    for (int j = 0; j < MR.nbcol; j++)
                    {
                        Mresult.data[i][j] += ML.data[i][k] * MR.data[k][j];

                    }
                }
            }
            return Mresult;
        }

        public static Vector Multiply(Matrix ML, Vector V)
        {
            Vector Vresult = new Vector(ML.nbrow);
            for (int i = 0; i < ML.nbrow; i++)
            {
                for (int k = 0; k < ML.nbcol; k++)
                {

                    Vresult.data[i] += ML.data[i][k] * V.data[k];


                }
            }

            return Vresult;
        }


    }


}
