
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OCRMNISTReader
{
    //This is the axon class (links between neuron)
    class AxonLayer
    {
        public NeuronLayer previousneuronlayer { get; set; }
        public NeuronLayer nextneuronlayer { get; set; }

        public Matrix weights { get; set; }

        public Matrix weightsupdate { get; set; }

        public static Parameters parameters { get; set; }

        //Constructor
        public AxonLayer(NeuronLayer previousneuronlayer, NeuronLayer nextneuronlayer)
        {
            this.previousneuronlayer = previousneuronlayer;
            this.nextneuronlayer = nextneuronlayer;
            this.weights = Matrix.Random(nextneuronlayer.nbneurons, previousneuronlayer.nbneurons, 1.0/Math.Sqrt(previousneuronlayer.nbneurons));
            this.weightsupdate = new Matrix(nextneuronlayer.nbneurons, previousneuronlayer.nbneurons);
        }

        //Reset the backpropagation matrix (to be done after each batch)
        public void Reset_Backprop()
        {
            this.weightsupdate.Clear();
        }

        //BACKPROPAGATION
        public void BackPropagate()
        {
            this.weightsupdate.UpdateBackProp(nextneuronlayer.delta, this.previousneuronlayer.neuron_activation);
        }

        //Update the weights, then reset
        public void Update_Weights(int batchsize)
        {
            this.weightsupdate.ScalarMultiply(-(parameters.eta / (double)batchsize));
            if (parameters.regularization)
            {
                this.weights.ScalarMultiply(1 - ((parameters.eta * parameters.lambda) / 50000.0)    );
            }

            this.weights.TermbyTermAdd(this.weightsupdate)    ;
            Reset_Backprop();
        }
    }
}
