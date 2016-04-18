using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OCRMNISTReader
{
    //This class only holds the parameters for a network
    public class Parameters
    {
        public int nblayers { get; set; }
        public int[] layers_size { get; set; }
        public double eta { get; set; }

        public int batchsize { get; set; }



        public enum Costfunction { SQUARE, CROSSENTROPY };
        public Costfunction costfunction { get; set; }

        public bool regularization { get; set; }
        public double lambda { get; set; }

        public int stopafternbsteps { get; set; }

        public Parameters()
        {

        }

        override public string ToString()
        {
            String s = "layers : " + string.Join(",", layers_size.ToArray()) ;
            s += " \n";
            s += "batchsize : " + batchsize + "  lambda : " + lambda;
            return s;

        }
    }
}
