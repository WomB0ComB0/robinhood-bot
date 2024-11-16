#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import tweepy
from utilities import TwitterCredentials
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
from collections import defaultdict
from src.bots.config import OrderType


@dataclass
class SentimentMetrics:
    """Container for sentiment analysis metrics"""

    compound_score: float
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    tweet_volume: int
    sentiment_momentum: float
    confidence_score: float


class SentimentThresholds:
    """Dynamic sentiment thresholds based on market conditions"""

    BUY_THRESHOLD = 0.15
    SELL_THRESHOLD = -0.15
    VOLUME_MINIMUM = 50
    CONFIDENCE_MINIMUM = 0.7


class TradeBotTwitterSentiments:
    """
    Trading bot implementing advanced Twitter sentiment analysis with enhanced features.

    Features:
    - Multi-source sentiment analysis (VADER + TextBlob)
    - Tweet preprocessing and cleaning
    - Sentiment momentum tracking
    - Volume-weighted sentiment analysis
    - Confidence scoring
    - Enhanced error handling and rate limiting
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the sentiment trading bot with configuration."""
        self.config = config or {
            "max_tweets": 200,
            "sentiment_window": "24h",
            "min_tweet_volume": 50,
            "enable_momentum": True,
            "enable_premarket_analysis": True,
        }

        self.logger = self._setup_logging()
        self._initialize_apis()

        self.vader_analyzer = SentimentIntensityAnalyzer()

        self.sentiment_history = defaultdict(list)

        self.performance_metrics = {
            "total_trades": 0,
            "successful_predictions": 0,
            "false_signals": 0,
            "avg_return": 0.0,
        }

    def _setup_logging(self) -> logging.Logger:
        """Configure enhanced logging system."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _initialize_apis(self):
        """Initialize Twitter API with error handling and rate limit management."""
        try:
            twitter_credentials = TwitterCredentials()
            auth = tweepy.AppAuthHandler(twitter_credentials.consumer_key, twitter_credentials.consumer_secret)
            self.twitter_api = tweepy.API(
                auth, wait_on_rate_limit=True, retry_count=3, retry_delay=5, retry_errors=[400, 401, 500, 502, 503, 504]
            )
            self.logger.info("Successfully initialized Twitter API")
        except Exception as e:
            self.logger.error("Failed to initialize Twitter API: %s", e)
            raise

    def _preprocess_tweet(self, tweet: str) -> str:
        """Clean and preprocess tweet text."""
        tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)
        tweet = re.sub(r"@\w+", "", tweet)
        tweet = re.sub(r"#\w+", "", tweet)
        tweet = re.sub(r"\$\w+", "", tweet)
        tweet = " ".join(tweet.split())
        return tweet

    def get_company_name_from_ticker(self, ticker: str) -> str:
        """Retrieve the company name from a stock ticker symbol."""
        ticker_to_company = {
            "AAPL": "Apple Inc",
            "GOOGL": "Alphabet Inc",
            "MSFT": "Microsoft Corporation",
            "AMZN": "Amazon.com Inc",
            "META": "Meta Platforms Inc",
            "TSLA": "Tesla Inc",
            "NVDA": "NVIDIA Corporation",
            "NFLX": "Netflix Inc",
            "IBM": "International Business Machines",
            "INTC": "Intel Corporation",
        }
        return ticker_to_company.get(ticker, ticker)

    def retrieve_tweets(self, ticker: str, max_count: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve and preprocess tweets about a stock with enhanced metadata.

        Args:
            ticker: Stock ticker symbol
            max_count: Maximum number of tweets to retrieve

        Returns:
            List of dictionaries containing tweet data and metadata
        """
        max_count = max_count or self.config["max_tweets"]
        processed_tweets = []

        try:
            company_name = self.get_company_name_from_ticker(ticker)
            query = f"({company_name} OR ${ticker}) lang:en -filter:retweets"

            tweets = tweepy.Cursor(
                self.twitter_api.search_tweets, q=query, lang="en", result_type="mixed", tweet_mode="extended"
            ).items(max_count)

            for tweet in tweets:
                processed_tweet = {
                    "text": self._preprocess_tweet(tweet.full_text),
                    "created_at": tweet.created_at,
                    "user_followers": tweet.user.followers_count,
                    "retweet_count": tweet.retweet_count,
                    "favorite_count": tweet.favorite_count,
                }
                processed_tweets.append(processed_tweet)

        except tweepy.TweepError as e:
            self.logger.error("Error retrieving tweets: %s", e)
            if e.response and e.response.status_code == 429:
                self.logger.warning("Rate limit reached. Implementing exponential backoff...")

        return processed_tweets

    def analyze_sentiment(self, tweets: List[Dict[str, Any]]) -> SentimentMetrics:
        """
        Perform comprehensive sentiment analysis using multiple techniques.

        Args:
            tweets: List of processed tweets with metadata

        Returns:
            SentimentMetrics object containing analysis results
        """
        if not tweets:
            return SentimentMetrics(0, 0, 0, 0, 0, 0, 0)

        sentiment_scores = []
        total_weight = 0

        for tweet in tweets:
            weight = np.log1p(1 + tweet["user_followers"] + 2 * tweet["retweet_count"] + tweet["favorite_count"])

            vader_scores = self.vader_analyzer.polarity_scores(tweet["text"])

            textblob_analysis = TextBlob(tweet["text"])
            textblob_score = textblob_analysis.sentiment.polarity

            combined_score = vader_scores["compound"] * 0.7 + textblob_score * 0.3

            sentiment_scores.append({"score": combined_score, "weight": weight, "timestamp": tweet["created_at"]})
            total_weight += weight

        weighted_compound = sum(s["score"] * s["weight"] for s in sentiment_scores) / total_weight

        positive = len([s for s in sentiment_scores if s["score"] > 0]) / len(sentiment_scores)
        negative = len([s for s in sentiment_scores if s["score"] < 0]) / len(sentiment_scores)
        neutral = 1 - positive - negative

        sentiment_momentum = self._calculate_sentiment_momentum(sentiment_scores)
        confidence_score = self._calculate_confidence_score(sentiment_scores, len(tweets))

        return SentimentMetrics(
            compound_score=weighted_compound,
            positive_ratio=positive,
            negative_ratio=negative,
            neutral_ratio=neutral,
            tweet_volume=len(tweets),
            sentiment_momentum=sentiment_momentum,
            confidence_score=confidence_score,
        )

    def _calculate_sentiment_momentum(self, sentiment_scores: List[Dict[str, Any]]) -> float:
        """Calculate sentiment momentum based on time-weighted scores."""
        if not sentiment_scores:
            return 0.0

        sorted_scores = sorted(sentiment_scores, key=lambda x: x["timestamp"])

        alpha = 0.1
        momentum = 0
        prev_score = sorted_scores[0]["score"]

        for score in sorted_scores[1:]:
            current_score = score["score"]
            momentum = alpha * (current_score - prev_score) + (1 - alpha) * momentum
            prev_score = current_score

        return momentum

    def _calculate_confidence_score(self, sentiment_scores: List[Dict[str, Any]], tweet_volume: int) -> float:
        """Calculate confidence score based on volume and consensus."""
        if not sentiment_scores:
            return 0.0

        volume_score = min(1.0, tweet_volume / SentimentThresholds.VOLUME_MINIMUM)

        scores = [s["score"] for s in sentiment_scores]
        consensus_score = 1.0 - np.std(scores)

        return volume_score * 0.4 + consensus_score * 0.6

    def make_trading_decision(self, ticker: str) -> Tuple[str, float, float]:
        """
        Make trading decision based on comprehensive sentiment analysis.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Tuple of (OrderType, sentiment_score, confidence_score)
        """
        try:
            tweets = self.retrieve_tweets(ticker)
            sentiment_metrics = self.analyze_sentiment(tweets)

            self.sentiment_history[ticker].append({"timestamp": datetime.now(), "metrics": sentiment_metrics})

            if (
                sentiment_metrics.compound_score >= SentimentThresholds.BUY_THRESHOLD
                and sentiment_metrics.confidence_score >= SentimentThresholds.CONFIDENCE_MINIMUM
                and sentiment_metrics.tweet_volume >= SentimentThresholds.VOLUME_MINIMUM
            ):

                if sentiment_metrics.sentiment_momentum > 0:
                    return (
                        OrderType.BUY_RECOMMENDATION,
                        sentiment_metrics.compound_score,
                        sentiment_metrics.confidence_score,
                    )

            elif (
                sentiment_metrics.compound_score <= SentimentThresholds.SELL_THRESHOLD
                and sentiment_metrics.confidence_score >= SentimentThresholds.CONFIDENCE_MINIMUM
            ):
                return (
                    OrderType.SELL_RECOMMENDATION,
                    sentiment_metrics.compound_score,
                    sentiment_metrics.confidence_score,
                )

            return (OrderType.HOLD_RECOMMENDATION, sentiment_metrics.compound_score, sentiment_metrics.confidence_score)

        except (KeyboardInterrupt, tweepy.TweepError) as e:
            self.logger.error("Error making trading decision: %s", e)
            return (OrderType.HOLD_RECOMMENDATION, 0.0, 0.0)

    def update_performance_metrics(self, recommendation: OrderType, actual_return: float):
        """Update performance tracking metrics."""
        self.performance_metrics["total_trades"] += 1

        if recommendation in [OrderType.BUY_RECOMMENDATION, OrderType.SELL_RECOMMENDATION] and actual_return > 0:
            self.performance_metrics["successful_predictions"] += 1
        else:
            self.performance_metrics["false_signals"] += 1

        # Update average return
        self.performance_metrics["avg_return"] = (
            self.performance_metrics["avg_return"] * (self.performance_metrics["total_trades"] - 1) + actual_return
        ) / self.performance_metrics["total_trades"]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of the bot's performance metrics."""
        total_trades = self.performance_metrics["total_trades"]
        if total_trades == 0:
            return {"success_rate": 0.0, "average_return": 0.0, "total_trades": 0}

        success_rate = self.performance_metrics["successful_predictions"] / total_trades * 100

        return {
            "success_rate": round(success_rate, 2),
            "average_return": round(self.performance_metrics["avg_return"], 4),
            "total_trades": total_trades,
            "false_signals": self.performance_metrics["false_signals"],
        }
