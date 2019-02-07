package com.example.demo.nets;

import java.io.File;
import java.util.Random;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.SimpleCNN;
import org.deeplearning4j.zoo.model.SimpleCNN.SimpleCNNBuilder;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.example.demo.RunTracker;

/**
 * Created by tom hanlon on 11/7/16. This code example is featured in this
 * youtube video https://www.youtube.com/watch?v=GLC8CIoHDnI
 *
 * This differs slightly from the Video Example The Video example had the data
 * already downloaded This example includes code that downloads the data
 *
 * The Data Directory mnist_png will have two child directories training and
 * testing The training and testing directories will have directories 0-9 with
 * 28 * 28 PNG images of handwritten images
 *
 * The code here shows how to use a ParentPathLabelGenerator to label the images
 * as they are read into the RecordReader
 *
 * The pixel values are scaled to values between 0 and 1 using
 * ImagePreProcessingScaler
 *
 * In this example a loop steps through 3 images and prints the DataSet to the
 * terminal. The expected output is the 28* 28 matrix of scaled pixel values the
 * list with the label for that image and a list of the label values
 *
 * This example also applies a Listener to the RecordReader that logs the path
 * of each image read You would not want to do this in production. The reason it
 * is done here is to show that a handwritten image 3 (for example) was read
 * from directory 3, has a matrix with the shown values, has a label value
 * corresponding to 3
 */
public class ConvNet {
	private static Logger log = LoggerFactory.getLogger(ConvNet.class);

	public void run(String basepath) throws Exception {
		if (RunTracker.getRuns().containsKey("deep") && RunTracker.getRuns().get("deep") == true) {
			return;
		}

		RunTracker.getRuns().put("deep", true);
		
		// Initialize the user interface backend
		UIServer uiServer = UIServer.getInstance();

		// Configure where the network information (gradients, score vs. time etc) is to
		// be stored. Here: store in memory.
		StatsStorage statsStorage = new InMemoryStatsStorage(); // Alternative: new FileStatsStorage(File), for saving
																// and loading later

		// Attach the StatsStorage instance to the UI: this allows the contents of the
		// StatsStorage to be visualized
		uiServer.attach(statsStorage);

		/*
		 * image information 28 * 28 grayscale grayscale implies single channel
		 */
		int height = 96;
		int width = 96;
		int channels = 1;
		int rngseed = new Random().nextInt();
		Random randNumGen = new Random(rngseed);
		int batchSize = 1;
		int outputNum = 20;

		/*
		 * This class downloadData() downloads the data stores the data in java's tmpdir
		 * 15MB download compressed It will take 158MB of space when uncompressed The
		 * data can be downloaded manually here
		 * http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz
		 */
//    downloadData();

		// Define the File Paths
		File trainData = new File(basepath + "/train");
		File testData = new File(basepath + "/test");

		// Define the FileSplit(PATH, ALLOWED FORMATS,random)
		FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
		FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

		// Extract the parent path as the image label
		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

		ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

		// Initialize the record reader
		// add a listener, to extract the name
		recordReader.initialize(train);

		// The LogRecordListener will log the path of each image read
		// used here for information purposes,
		// If the whole dataset was ingested this would place 60,000
		// lines in our logs
		// It will show up in the output with this format
		// o.d.a.r.l.i.LogRecordListener - Reading /tmp/mnist_png/training/4/36384.png
		recordReader.setListeners(new LogRecordListener());

		// DataSet Iterator
		DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1,
				recordReader.getLabels().size());

		// Scale pixel values to 0-1
		DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
		scaler.fit(dataIter);
		dataIter.setPreProcessor(scaler);

		String trainingLabels = dataIter.getLabels().toString();

		// In production you would loop through all the data
		// in this example the loop is just through 3
		// images for demonstration purposes
//    for (int i = 1; i < 3; i++) {
//      DataSet ds = dataIter.next();
//      log.info(ds.toString());
//      log.info(dataIter.getLabels().toString());
//    }
		log.info("BUILD MODEL");

		int[] shape = { channels, width, height };
		MultiLayerNetwork model = new MultiLayerNetwork(conf(outputNum));

		// The Score iteration Listener will log
		// output to show how well the network is training
		model.setListeners(new ScoreIterationListener(10), new StatsListener(statsStorage));

		log.info("TRAIN MODEL");
		for (int i = 0; i < 2; i++) {
			model.fit(dataIter);
		}

		log.info("EVALUATE MODEL");
		recordReader.reset();

		// The model trained on the training dataset split
		// now that it has trained we evaluate against the
		// test data of images the network has not seen

		recordReader.initialize(test);
		DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
		scaler.fit(testIter);
		testIter.setPreProcessor(scaler);

		/*
		 * log the order of the labels for later use In previous versions the label
		 * order was consistent, but random In current verions label order is
		 * lexicographic preserving the RecordReader Labels order is no longer needed
		 * left in for demonstration purposes
		 */
		String testlabels = recordReader.getLabels().toString();

		// Create Eval object with 10 possible classes
		Evaluation eval = new Evaluation(outputNum);

		// Evaluate the network
		while (testIter.hasNext()) {
			DataSet next = testIter.next();
			INDArray output = model.output(next.getFeatures());
			// Compare the Feature Matrix from the model
			// with the labels from the RecordReader
			eval.eval(next.getLabels(), output);
		}
		
		ModelSerializer.writeModel(model, new File("convnet-model.zip"), true);

		log.info("eval stats " + eval.stats());

		log.info("traininglabels: " + trainingLabels);
		log.info("testlabels: " + testlabels);
		RunTracker.getRuns().put("deep", false);

	}

	public static MultiLayerConfiguration conf(int numClasses) {

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(System.currentTimeMillis())
				.activation(Activation.IDENTITY).weightInit(WeightInit.RELU)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Nesterovs(0.005, 0.9))
				.cacheMode(CacheMode.NONE).trainingWorkspaceMode(WorkspaceMode.ENABLED)
				.inferenceWorkspaceMode(WorkspaceMode.ENABLED).convolutionMode(ConvolutionMode.Same).list()
				// block 1
				.layer(0,
						new ConvolutionLayer.Builder(new int[] { 7, 7 }).name("image_array").nIn(1).nOut(16)
								.build())
				.layer(1, new BatchNormalization.Builder().build())
				.layer(2, new ConvolutionLayer.Builder(new int[] { 7, 7 }).nIn(16).nOut(16).build())
				.layer(3, new BatchNormalization.Builder().build())
				.layer(4, new ActivationLayer.Builder().activation(Activation.RELU).build())
				.layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[] { 2, 2 }).build())
				.layer(6, new DropoutLayer.Builder(0.5).build())

				// block 2
				.layer(7, new ConvolutionLayer.Builder(new int[] { 5, 5 }).nOut(32).build())
				.layer(8, new BatchNormalization.Builder().build())
				.layer(9, new ConvolutionLayer.Builder(new int[] { 5, 5 }).nOut(32).build())
				.layer(10, new BatchNormalization.Builder().build())
				.layer(11, new ActivationLayer.Builder().activation(Activation.RELU).build())
				.layer(12, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[] { 2, 2 }).build())
				.layer(13, new DropoutLayer.Builder(0.5).build())

				// block 3
				.layer(14, new ConvolutionLayer.Builder(new int[] { 3, 3 }).nOut(64).build())
				.layer(15, new BatchNormalization.Builder().build())
				.layer(16, new ConvolutionLayer.Builder(new int[] { 3, 3 }).nOut(64).build())
				.layer(17, new BatchNormalization.Builder().build())
				.layer(18, new ActivationLayer.Builder().activation(Activation.RELU).build())
				.layer(19, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[] { 2, 2 }).build())
				.layer(20, new DropoutLayer.Builder(0.5).build())

				// block 4
				.layer(21, new ConvolutionLayer.Builder(new int[] { 3, 3 }).nOut(128).build())
				.layer(22, new BatchNormalization.Builder().build())
				.layer(23, new ConvolutionLayer.Builder(new int[] { 3, 3 }).nOut(128).build())
				.layer(24, new BatchNormalization.Builder().build())
				.layer(25, new ActivationLayer.Builder().activation(Activation.RELU).build())
				.layer(26, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[] { 2, 2 }).build())
				.layer(27, new DropoutLayer.Builder(0.5).build())

				// block 5
				.layer(28, new ConvolutionLayer.Builder(new int[] { 3, 3 }).nOut(256).build())
				.layer(29, new BatchNormalization.Builder().build())
				.layer(30, new ConvolutionLayer.Builder(new int[] { 3, 3 }).nOut(numClasses).build())
				.layer(31, new GlobalPoolingLayer.Builder(PoolingType.AVG).build())
				.layer(32, new ActivationLayer.Builder().activation(Activation.SOFTMAX).build())
				.layer(33,
						new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
								.activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER).name("Output")
								.nOut(numClasses).build())

				.setInputType(InputType.convolutional(96, 96, 1)).build();

		return conf;
	}
}
