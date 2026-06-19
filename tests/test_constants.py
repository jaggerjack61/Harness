"""Tests for centralized constants."""

from harness import constants


class TestConstants:
    """Verify all magic values are defined in one place with correct defaults."""

    def test_max_turns_default(self):
        assert constants.DEFAULT_MAX_TURNS == 1000

    def test_context_window_default(self):
        assert constants.DEFAULT_CONTEXT_WINDOW == 1000000

    def test_bash_timeout(self):
        assert constants.BASH_TIMEOUT == 60

    def test_max_output_lines(self):
        assert constants.MAX_OUTPUT_LINES == 1000

    def test_max_output_bytes(self):
        assert constants.MAX_OUTPUT_BYTES > 0

    def test_default_model(self):
        assert constants.DEFAULT_MODEL == "deepseek-v4-pro"

    def test_default_base_url(self):
        assert constants.DEFAULT_BASE_URL == "https://api.openai.com/v1"

    def test_default_reasoning_effort(self):
        assert constants.DEFAULT_REASONING_EFFORT == "high"

    def test_reasoning_options(self):
        assert constants.REASONING_OPTIONS == ["low", "medium", "high", "xhigh", "max"]

    def test_nonstandard_reasoning_efforts(self):
        assert constants.NONSTANDARD_REASONING_EFFORTS == {"xhigh", "max"}

    def test_max_retries(self):
        assert constants.MAX_RETRIES == 3

    def test_retry_base_delay(self):
        assert constants.RETRY_BASE_DELAY == 1.0

    def test_fetch_timeout(self):
        assert constants.FETCH_TIMEOUT == 10.0

    def test_display_truncation_limit(self):
        assert constants.DISPLAY_TRUNCATION_LIMIT == 5

    def test_live_refresh_streaming(self):
        assert constants.LIVE_REFRESH_STREAMING == 12

    def test_live_refresh_non_streaming(self):
        assert constants.LIVE_REFRESH_NON_STREAMING == 4

    def test_spinner_frames(self):
        assert len(constants.SPINNER_FRAMES) == 10

    def test_context_window_trim_threshold(self):
        assert 0 < constants.CONTEXT_WINDOW_TRIM_THRESHOLD < 1

    def test_recent_turns_to_keep(self):
        assert constants.RECENT_TURNS_TO_KEEP >= 2

    def test_summary_max_tokens(self):
        assert constants.SUMMARY_MAX_TOKENS > 0
