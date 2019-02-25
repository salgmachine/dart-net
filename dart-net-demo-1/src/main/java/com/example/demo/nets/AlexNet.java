package com.example.demo.nets;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.EarlyTerminationDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.Darknet19;
import org.deeplearning4j.zoo.model.SimpleCNN;
import org.deeplearning4j.zoo.model.VGG19;
import org.deeplearning4j.zoo.model.VGG19.VGG19Builder;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.convert.DataSizeUnit;

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
public class AlexNet {
	private static Logger log = LoggerFactory.getLogger(AlexNet.class);

	public void run(String basepath) throws Exception {

		if (RunTracker.getRuns().containsKey("alexnet") && RunTracker.getRuns().get("alexnet") == true) {
			return;
		}

		RunTracker.getRuns().put("alexnet", true);

		// Initialize the user interface backend
		UIServer uiServer = UIServer.getInstance();

		// Configure where the network information (gradients, score vs. time etc) is to
		// be stored. Here: store in memory.
		StatsStorage statsStorage = new InMemoryStatsStorage(); // Alternative: new FileStatsStorage(File), for saving
																// and loading later

		// Attach the StatsStorage instance to the UI: this allows the contents of the
		// StatsStorage to be visualized
		uiServer.attach(statsStorage);

		int height = 96;
		int width = 96;
		int channels = 1;
		int rngseed = 123;
		Random randNumGen = new Random(rngseed);
		int batchSize = 16;
		int outputNum = 20;

		// Define the File Paths
		File modelData = new File(basepath + "/model");
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

//		EarlyTerminationDataSetIterator early = new EarlyTerminationDataSetIterator(dataIter, 10000);

		// Scale pixel values to 0-1
		DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
		scaler.fit(dataIter);
		dataIter.setPreProcessor(scaler);

		String trainingLabels = dataIter.getLabels().toString();
		log.info("BUILD MODEL");

		MultiLayerNetwork model = new MultiLayerNetwork(conf(rngseed, channels, outputNum, height, width));
		model.init();

//		Path mdlPath = Paths.get(modelData + "/deep-model.json");
//		if (!Files.exists(mdlPath)) {
//			Files.createFile(mdlPath);
//		}
////		ModelSerializer.writeModel(model, modelData + "/deep-model.json", true);

		// The Score iteration Listener will log
		// output to show how well the network is training
		model.setListeners(new ScoreIterationListener(5), new StatsListener(statsStorage));

		log.info("TRAIN MODEL");
		model.fit(dataIter, 2);

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
		ModelSerializer.writeModel(model, new File("alexnet-model.zip"), true);

		log.info("eval stats " + eval.stats());

		log.info("traininglabels: " + trainingLabels);
		log.info("testlabels: " + testlabels);

		RunTracker.getRuns().put("alexnet", false);

	}

	public static MultiLayerConfiguration linear(long seed, int channels, int numLabels, int height, int width) {
		int nIn = channels * height * width;
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Nesterovs(0.1, 0.9))
				.list()
				.layer(0,
						new DenseLayer.Builder().nIn(nIn).nOut(nIn * 4).weightInit(WeightInit.XAVIER)
								.activation(Activation.RELU).build())
				.layer(1,
						new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).weightInit(WeightInit.XAVIER)
								.activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER).nIn(nIn * 4)
								.nOut(numLabels).build())
				.pretrain(false).backprop(true).build();
		return conf;
	}

	public static MultiLayerConfiguration mnist(long seed, int channels, int numLabels, int height, int width) {
		Map<Integer, Double> lrSchedule = new HashMap<>();
		lrSchedule.put(0, 0.06); // iteration #, learning rate
		lrSchedule.put(200, 0.05);
		lrSchedule.put(600, 0.028);
		lrSchedule.put(800, 0.0060);
		lrSchedule.put(1000, 0.001);

		return new NeuralNetConfiguration.Builder().seed(seed).l2(0.0005)
				// .learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
				.weightInit(WeightInit.XAVIER).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(new Nesterovs(0.1, 0.9)).list().layer(0, new ConvolutionLayer.Builder(5, 5)
						// nIn and nOut specify depth. nIn here is the nChannels and nOut is the number
						// of filters to be applied
						.nIn(channels).stride(1, 1).nOut(20).activation(Activation.IDENTITY).build())
				.layer(1,
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build())
				.layer(2, new ConvolutionLayer.Builder(5, 5)
						// Note that nIn need not be specified in later layers
						.stride(1, 1).nOut(50).activation(Activation.IDENTITY).build())
				.layer(3,
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build())
				.layer(4, new DenseLayer.Builder().activation(Activation.RELU).nOut(500).build())
				.layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(numLabels)
						.activation(Activation.SOFTMAX).build())
				.backprop(true).pretrain(false).build();
	}

	public static MultiLayerConfiguration conf(long seed, int channels, int numLabels, int height, int width) {

		int numClasses = 8;

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(System.currentTimeMillis())
				.activation(Activation.IDENTITY).weightInit(WeightInit.RELU)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Nesterovs(0.0001,0.9))
				.cacheMode(CacheMode.NONE).trainingWorkspaceMode(WorkspaceMode.ENABLED)
				.inferenceWorkspaceMode(WorkspaceMode.ENABLED).convolutionMode(ConvolutionMode.Same).list()
				// block 1
				.layer(0,
						new ConvolutionLayer.Builder(new int[] { 7, 7 }).name("image_array").nIn(channels).nOut(16)
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
								.nOut(numLabels).build())

				.setInputType(InputType.convolutionalFlat(height, width, channels)).build();

		return conf;
	}

	public static MultiLayerConfiguration simple(long seed, int channels, int numLabels, int height, int width) {
		return new NeuralNetConfiguration.Builder().seed(seed)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(new Nesterovs(0.2, 0.9)) // High
																														// Level
																														// Configuration
				.list() // For configuring MultiLayerNetwork we call the list method
				.layer(0,
						new DenseLayer.Builder().nIn(channels * height * width).nOut((channels * height * width) * 4).weightInit(WeightInit.XAVIER)
								.activation(Activation.RELU).build()) // Configuring Layers
				.layer(1,
						new OutputLayer.Builder().nOut(numLabels).weightInit(WeightInit.XAVIER)
								.activation(Activation.SOFTMAX).build())
				.pretrain(false).backprop(true) // Pretraining and Backprop Configuration
				.build(); // Building Configuration
	}

	public static MultiLayerConfiguration alexnetModel(long seed, int channels, int numLabels, int height, int width) {
		/**
		 * AlexNet model interpretation based on the original paper ImageNet
		 * Classification with Deep Convolutional Neural Networks and the
		 * imagenetExample code referenced.
		 * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
		 **/

		double nonZeroBias = 1;
		double dropOut = 0.5;

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().cudnnAlgoMode(AlgoMode.NO_WORKSPACE)
				.seed(seed).weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0.0, 0.01))
				.activation(Activation.RELU)
				.updater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 1e-2, 0.1, 100000), 0.9))
				.biasUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 2e-2, 0.1, 100000), 0.9))
				.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or
																					// exploding gradients
				.l2(5 * 1e-4).list()
				.layer(0,
						convInit("cnn1", channels, 96, new int[] { 11, 11 }, new int[] { 4, 4 }, new int[] { 3, 3 }, 0))
				.layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
				.layer(2, maxPool("maxpool1", new int[] { 3, 3 }))
				.layer(3, conv5x5("cnn2", 256, new int[] { 1, 1 }, new int[] { 2, 2 }, nonZeroBias))
				.layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
				.layer(5, maxPool("maxpool2", new int[] { 3, 3 })).layer(6, conv3x3("cnn3", 384, 0))
				.layer(7, conv3x3("cnn4", 384, nonZeroBias)).layer(8, conv3x3("cnn5", 256, nonZeroBias))
				.layer(9, maxPool("maxpool3", new int[] { 3, 3 }))
				.layer(10, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
				.layer(11, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
				.layer(12,
						new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).name("output")
								.nOut(numLabels).activation(Activation.SOFTMAX).build())
				.backprop(true).pretrain(false).setInputType(InputType.convolutional(height, width, channels)).build();

		return conf;

	}

	private static ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad,
			double bias) {
		return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
	}

	private static ConvolutionLayer conv3x3(String name, int out, double bias) {
		return new ConvolutionLayer.Builder(new int[] { 3, 3 }, new int[] { 1, 1 }, new int[] { 1, 1 }).name(name)
				.nOut(out).biasInit(bias).build();
	}

	private static ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
		return new ConvolutionLayer.Builder(new int[] { 5, 5 }, stride, pad).name(name).nOut(out).biasInit(bias)
				.build();
	}

	private static SubsamplingLayer maxPool(String name, int[] kernel) {
		return new SubsamplingLayer.Builder(kernel, new int[] { 2, 2 }).name(name).build();
	}

	private static DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
		return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build();
	}

}
