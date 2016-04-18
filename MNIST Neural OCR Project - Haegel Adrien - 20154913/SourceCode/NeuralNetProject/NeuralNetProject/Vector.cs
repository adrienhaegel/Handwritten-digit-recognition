using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OCRMNISTReader
{

    //A vector implementation
    class Vector
    {

        public double[] data;
        public int size;

        public Vector(int size)
        {
            data = new double[size];
            this.size = size;
            /*
            for(int i=0;i< size; i++)
            {
                data[i] = 0;
            }
            */
        }



        public static Vector Random(int nbrow, double deviation)
        {
            Vector m = new Vector(nbrow);

            Random rand = new Random(); //reuse this if you are generating many


            for (int i = 0; i < nbrow; i++)
            {

                double u1 = rand.NextDouble(); //these are uniform(0,1) random doubles
                double u2 = rand.NextDouble();
                double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                             Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
                double randNormal =
                             0 + deviation * randStdNormal; //random normal(mean,stdDev^2)


                m.data[i] = randNormal;

            }

            return m;
        }

        public void Clear()
        {
            for (int i = 0; i < size; i++)
            {

                data[i] = 0;

            }
        }

        public double Get(int index)
        {
            return this.data[index];
        }



        public void ScalarMultiply(double scalar)
        {
            for (int i = 0; i < size; i++)
            {

                data[i] *= scalar;

            }
        }

        public void ScalarAddition(double scalar)
        {
            for (int i = 0; i < size; i++)
            {

                data[i] += scalar;

            }
        }


        public void TermbyTermAdd(Vector Vadd)
        {
            for (int i = 0; i < size; i++)
            {

                data[i] += Vadd.data[i];

            }
        }





        public Vector Map(Func<double, double> func)
        {
            Vector Vresult = new Vector(this.size);
            for (int i = 0; i < size; i++)
            {

                Vresult.data[i] = func(this.data[i]);

            }
            return Vresult;
        }


        public int MaximumIndex()
        {
            int index = 0;
            double value = this.data[0];

            for (int i = 1; i < this.size; i++)
            {
                if (this.data[i] > value)
                {
                    value = this.data[i];
                    index = i;
                }
            }
            return index;
        }



        //STATIC METHODS

        public static Vector TermbyTermAdd(Vector VL, Vector VR)
        {
            Vector Vresult = new Vector(VL.size);
            for (int i = 0; i < VL.size; i++)
            {

                Vresult.data[i] = VL.data[i] + VR.data[i];

            }
            return Vresult;
        }


        public static Vector TermbyTermSubtract(Vector VL, Vector VR)
        {
            Vector Vresult = new Vector(VL.size);
            for (int i = 0; i < VL.size; i++)
            {

                Vresult.data[i] = VL.data[i] - VR.data[i];

            }
            return Vresult;
        }


        public static Vector TermbyTermMultiply(Vector VL, Vector VR)
        {
            Vector Vresult = new Vector(VL.size);
            for (int i = 0; i < VL.size; i++)
            {

                Vresult.data[i] = VL.data[i] * VR.data[i];

            }
            return Vresult;
        }

        public static Vector ScalarMultiply(Vector V, double scalar)
        {
            Vector Vresult = new Vector(V.size);
            for (int i = 0; i < V.size; i++)
            {

                Vresult.data[i] = V.data[i] * scalar;

            }
            return Vresult;
        }

        public static Matrix VectorTVectorMultiply(Vector VL, Vector VR)
        {
            Matrix Mresult = new Matrix(VL.size, VR.size);
            for(int i = 0; i < VL.size; i++)
            {
                Mresult.data[i] = Vector.ScalarMultiply(VR,VL.data[i]).data;
            }
            return Mresult;
        }

    }

}
