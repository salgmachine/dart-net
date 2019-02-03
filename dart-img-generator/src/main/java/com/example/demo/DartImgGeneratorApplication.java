package com.example.demo;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

import com.google.common.collect.Lists;

import io.reactivex.Observable;
import io.reactivex.schedulers.Schedulers;

@SpringBootApplication
public class DartImgGeneratorApplication {

	// degree offset for next number: 2 degrees
	// degree offset for left to right: 16 degrees
	public static final List<Run> runs = Arrays.asList(new Run("20", -8, -8 + 16), new Run("3", 10, 10 + 16),
			new Run("16", 28, 28 + 16), new Run("7", 46, 46 + 16), new Run("12", 64, 64 + 16),
			new Run("11", 82, 82 + 16), new Run("8", 100, 100 + 16), new Run("15", 118, 118 + 16),
			new Run("4", 136, 136 + 16), new Run("19", 154, 154 + 16), new Run("2", 172, 172 + 16),
			new Run("17", 190, 190 + 16), new Run("6", 208, 208 + 16), new Run("13", 226, 226 + 16),
			new Run("10", 244, 244 + 16), new Run("9", 262, 262 + 16), new Run("14", 280, 280 + 16),
			new Run("5", 298, 298 + 16), new Run("18", 314, 314 + 16), new Run("1", 332, 332 + 16)
//			
	);

	public static void main(String[] args) {
		ConfigurableApplicationContext ctx = SpringApplication.run(DartImgGeneratorApplication.class, args);
		Loader bean = ctx.getBean(Loader.class);

		List<Observable<Run>> observableRuns = runs.stream().map(run -> {
			return Observable.fromCallable(() -> {
				return run.execute(bean);
			}).subscribeOn(Schedulers.newThread());
		}).collect(Collectors.toList());

		Lists.partition(observableRuns, 8).forEach(batch -> Observable.zip(batch, (n) -> Boolean.TRUE).blockingFirst());

	}

	@Component
	public class ImgWrittenEventListener {

		private final Logger log = LoggerFactory.getLogger(getClass());

		@Autowired
		private Counter counter;

		@EventListener
		public void onImgWritten(ImgWrittenEvent evt) {

			Integer count = counter.increment();
			Boolean isTestImg = evt.getIsTestImg();
			String label = evt.getLabel();
			counter.incrementForLabel(label, isTestImg);

			Integer trainingValues = counter.getTrainingLabelCounts().values().stream().mapToInt(Integer::intValue).sum();
			Integer testingValues = counter.getTestingLabelCounts().values().stream().mapToInt(Integer::intValue).sum();
			

			if (count % 50 == 0) {
				log.info("Generated {} images so far", count);
				log.info("Training Data ({}) : {}", trainingValues, counter.getTrainingLabelCounts());
				log.info("Testing Data  ({}) : {}", testingValues, counter.getTestingLabelCounts());
			}
		}
	}
}
