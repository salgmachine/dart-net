package com.example.demo;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.EnableAsync;

@EnableAsync
@Configuration
public class AppConfig {

//	@Bean
//	public UIServer uiServer() {
//		UIServer uiServer = UIServer.getInstance();
//		uiServer.attach(statsStorage());
//		return uiServer;
//	}
//
//	@Bean
//	public StatsStorage statsStorage() {
//		return new InMemoryStatsStorage();
//	}

}
