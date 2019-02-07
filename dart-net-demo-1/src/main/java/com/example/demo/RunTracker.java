package com.example.demo;

import java.util.HashMap;
import java.util.Map;

public class RunTracker {

	private static final Map<String, Boolean> runs = new HashMap<String, Boolean>();

	public static Map<String, Boolean> getRuns() {
		return runs;
	}
}
