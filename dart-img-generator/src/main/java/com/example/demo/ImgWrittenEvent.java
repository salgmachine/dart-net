package com.example.demo;

import java.nio.file.Path;
import java.util.Date;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class ImgWrittenEvent {

	private final Path path;
	private final Date writtenAt;

	private final Boolean isTestImg;
	private final String label;

}
