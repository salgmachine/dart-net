package com.example.demo;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Data
public class Run {
	private final String prefix;
	private final int rotateMin;
	private final int rotateMax;

	public Run(String prefix, int rotateMin, int rotateMax) {
		this.prefix = prefix;
		this.rotateMin = rotateMin;
		this.rotateMax = rotateMax;
	}

	public Run execute(Loader loader) {
		loader.addWatermark(getPrefix(), getRotateMin(), getRotateMax());
		return this;
	}

}