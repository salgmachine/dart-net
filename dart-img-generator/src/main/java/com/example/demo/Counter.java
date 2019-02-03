package com.example.demo;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

import org.springframework.stereotype.Component;

@Component
public class Counter {

	private Integer count = 0;

	private final Map<Integer, Integer> trainingLabelCounts = new HashMap<>();
	private final Map<Integer, Integer> testingLabelCounts = new HashMap<>();

	public Counter() {
		IntStream.rangeClosed(1, 20).forEach(key -> {
			trainingLabelCounts.put(key, 0);
			testingLabelCounts.put(key, 0);
		});
	}

	public Map<Integer, Integer> getTrainingLabelCounts() {
		return trainingLabelCounts;
	}

	public Map<Integer, Integer> getTestingLabelCounts() {
		return testingLabelCounts;
	}

	public Integer incrementForLabel(String label, boolean isTesting) {
		Integer i = Integer.parseInt(label);

		Integer integer = (isTesting ? testingLabelCounts.get(i) : trainingLabelCounts.get(i));
		integer = integer + 1;
		if (isTesting) {
			testingLabelCounts.put(i, integer);
		} else {
			trainingLabelCounts.put(i, integer);
		}
		return integer;
	}

	public Integer increment() {
		this.count = this.count + 1;
		return this.count;
	}

	public Integer getCount() {
		return this.count;
	}

}
