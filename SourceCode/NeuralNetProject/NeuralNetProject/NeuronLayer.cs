
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OCRMNISTReader
{
    //Class for a neuron layer
    class NeuronLayer
    {
        NeuronLayer previousneuronlayer { get; set; }
        NeuronLayer nextneuronlayer { get; set; }

        AxonLayer previousaxonlayer { get; set; }
        AxonLayer nextaxonlayer { get; set; }

        public int nbneurons { get; set; }

        public static Parameters parameters { get; set; }


        Vector neuron_z;
        public Vector neuron_activation;
        Vector neuron_bias;

        public Vector delta { get; set; }
        public Vector biasupdate { get; set; }


        //constructor for the first layer (input)
        public NeuronLayer(int nbneurons)
        {
            neuron_activation = Vector.Random(nbneurons, 1);
            neuron_bias = Vector.Random(nbneurons, 1);
            this.nbneurons = nbneurons;
        }

        //constructor for the other layers (builds all the links with the previous layer)
        public NeuronLayer(int nbneurons, NeuronLayer previousneuronlayer)
        {
            neuron_activation = new Vector(nbneurons);
            neuron_bias = Vector.Random(nbneurons, 1);
            this.biasupdate = new Vector(nbneurons);
            this.nbneurons = nbneurons;

            this.previousneuronlayer = previousneuronlayer;
            previousneuronlayer.nextneuronlayer = this;

            AxonLayer axonlayer = new AxonLayer(this.previousneuronlayer, this);
            this.previousaxonlayer = axonlayer;
            previousneuronlayer.nextaxonlayer = axonlayer;
        }


        //Reset the bias update sum
        public void Reset_Backprop_Neurons()
        {
            this.biasupdate.Clear();
        }


        //Forward propagation
        public void FeedForward()
        {
            this.neuron_z = Matrix.Forward_neuronz(this.previousaxonlayer.weights, this.previousneuronlayer.neuron_activation, this.neuron_bias);
            this.neuron_activation = this.neuron_z.Map(sigmoid);
        }

        //Set values for the first layer (input data)
        public void SetActivation(Vector values)
        {
            this.neuron_activation = values;
        }

        //sigmoid function and its derivative
        public double sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(x * (-1)));
        }
        public double sigmoidprim(double x)
        {
            return sigmoid(x) * (1 - sigmoid(x));
        }


        //Return the ouptut guess 
        public int GetResult()
        {
            return this.neuron_activation.MaximumIndex();
        }

        //Compute the delta(L)
        public void Compute_Output_Error(Vector output)
        {
            if (parameters.costfunction == Parameters.Costfunction.SQUARE)
            {
                this.delta =   Vector.TermbyTermMultiply(Vector.TermbyTermSubtract(this.neuron_activation, output) ,     this.neuron_z.Map(sigmoidprim)    );
            }
            else if (parameters.costfunction == Parameters.Costfunction.CROSSENTROPY)
            {
                this.delta = Vector.TermbyTermSubtract(this.neuron_activation, output);
            }

            else
            {
                throw new Exception();
            }

            this.biasupdate.TermbyTermAdd(delta);

            this.previousaxonlayer.BackPropagate();
        }

        //Backpropagation
        public void Backpropagate()
        {
            this.delta = Vector.TermbyTermMultiply(Matrix.Multiply(this.nextaxonlayer.weights.Transpose(), this.nextneuronlayer.delta), this.neuron_z.Map(sigmoidprim));

            this.biasupdate.TermbyTermAdd(delta);

            //Call the propagation backwards
            this.previousaxonlayer.BackPropagate();

        }

        //Update after the end of a batch
        public void Update_Bias_and_Weights(int batchsize)
        {
            this.biasupdate.ScalarMultiply(-(parameters.eta / (double)batchsize));
            this.neuron_bias.TermbyTermAdd(this.biasupdate)   ;
            Reset_Backprop_Neurons();
            if (this.previousaxonlayer != null)
            {
                this.previousaxonlayer.Update_Weights(batchsize);
            }
        }

    }
}
