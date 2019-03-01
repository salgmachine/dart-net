package com.example.demo;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Data
public class Run {
	private final String prefix;
	private final int rotateMin;
	private final int rotateMax;
	private int margin = 0;

	public Run(String prefix, int rotateMin, int rotateMax) {
		this.prefix = prefix;
		this.rotateMin = (rotateMin < 0 ? rotateMin + margin : rotateMin - margin);
		this.rotateMax = (rotateMax < 0 ? rotateMax + margin : rotateMax - margin);
	}

	public Run execute(Loader loader) {
		loader.addWatermark(getPrefix(), getRotateMin(), getRotateMax());
		return this;
	}

}