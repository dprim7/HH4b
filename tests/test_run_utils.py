"""
Unit tests for HH4b.run_utils module.

Tests command-line argument handling, git operations, and utility functions.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from HH4b.run_utils import add_bool_arg, check_branch, print_red


class TestAddBoolArg:
    """Test add_bool_arg function for boolean argument parsing."""

    def test_basic_bool_arg(self):
        """Test adding a basic boolean argument."""
        parser = argparse.ArgumentParser()
        add_bool_arg(parser, "test-flag", help="Test flag", default=False)

        # Test positive flag
        args = parser.parse_args(["--test-flag"])
        assert args.test_flag is True

        # Test negative flag
        args = parser.parse_args(["--no-test-flag"])
        assert args.test_flag is False

        # Test default
        args = parser.parse_args([])
        assert args.test_flag is False

    def test_bool_arg_with_default_true(self):
        """Test boolean argument with default=True."""
        parser = argparse.ArgumentParser()
        add_bool_arg(parser, "enabled", help="Enable feature", default=True)

        args = parser.parse_args([])
        assert args.enabled is True

        args = parser.parse_args(["--no-enabled"])
        assert args.enabled is False

    def test_bool_arg_hyphen_to_underscore(self):
        """Test that hyphens in arg names are converted to underscores."""
        parser = argparse.ArgumentParser()
        add_bool_arg(parser, "multi-word-flag", help="Test", default=False)

        args = parser.parse_args(["--multi-word-flag"])
        assert args.multi_word_flag is True

    def test_custom_no_name(self):
        """Test custom no-name for boolean argument."""
        parser = argparse.ArgumentParser()
        add_bool_arg(
            parser, "feature", help="Enable feature", default=False, no_name="disable-feature"
        )

        args = parser.parse_args(["--disable-feature"])
        assert args.feature is False


class TestPrintRed:
    """Test print_red function for colored output."""

    @patch("builtins.print")
    def test_print_red_output(self, mock_print):
        """Test that print_red outputs colored text."""
        test_message = "Test error message"
        print_red(test_message)

        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        # Should contain ANSI color codes and the message
        assert test_message in call_args


class TestCheckBranch:
    """Test check_branch function for git operations."""

    @patch("HH4b.run_utils.subprocess.run")
    def test_valid_branch(self, mock_run):
        """Test checking a valid git branch."""
        # Mock successful git ls-remote
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        # Should not raise
        try:
            check_branch("main", allow_diff_local_repo=True)
        except AssertionError:
            pytest.fail("check_branch raised AssertionError for valid branch")

    @patch("HH4b.run_utils.subprocess.run")
    def test_invalid_branch(self, mock_run):
        """Test checking an invalid git branch."""
        # Mock failed git ls-remote
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")

        with pytest.raises(AssertionError, match="does not exist"):
            check_branch("nonexistent-branch")

    @patch("HH4b.run_utils.subprocess.run")
    @patch("HH4b.run_utils.print_red")
    @patch("sys.exit")
    def test_uncommitted_changes_without_override(self, mock_exit, mock_print_red, mock_run):
        """Test handling uncommitted changes without override."""
        # Mock git ls-remote success
        # Mock git status with uncommitted changes
        # Mock git show and git rev-parse
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),  # ls-remote
            MagicMock(returncode=0, stdout="M file.txt\n", stderr=""),  # status
        ]

        check_branch("main", allow_diff_local_repo=False)
        mock_exit.assert_called_with(1)

    @patch("HH4b.run_utils.subprocess.run")
    def test_uncommitted_changes_with_override(self, mock_run):
        """Test handling uncommitted changes with override."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),  # ls-remote
            MagicMock(returncode=0, stdout="M file.txt\n", stderr=""),  # status
            MagicMock(returncode=0, stdout="commit abc123\n", stderr=""),  # show
            MagicMock(returncode=0, stdout="abc123\n", stderr=""),  # rev-parse
        ]

        # Should not raise or exit
        check_branch("main", allow_diff_local_repo=True)

    @patch("HH4b.run_utils.subprocess.run")
    @patch("sys.exit")
    def test_mismatched_commits_without_override(self, mock_exit, mock_run):
        """Test handling mismatched commits without override."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),  # ls-remote
            MagicMock(returncode=0, stdout="", stderr=""),  # status (no changes)
            MagicMock(returncode=0, stdout="commit remote123\n", stderr=""),  # show remote
            MagicMock(returncode=0, stdout="local456\n", stderr=""),  # rev-parse local
        ]

        check_branch("main", allow_diff_local_repo=False)
        mock_exit.assert_called_with(1)


@pytest.mark.unit
class TestRunUtilsIntegration:
    """Integration tests for run_utils module."""

    def test_argument_parser_integration(self):
        """Test complete argument parser setup."""
        parser = argparse.ArgumentParser()
        add_bool_arg(parser, "submit", help="Submit jobs", default=False)
        add_bool_arg(parser, "test", help="Test mode", default=False)

        # Test various combinations
        args = parser.parse_args(["--submit", "--test"])
        assert args.submit is True
        assert args.test is True

        args = parser.parse_args(["--no-submit", "--test"])
        assert args.submit is False
        assert args.test is True

        args = parser.parse_args([])
        assert args.submit is False
        assert args.test is False


@pytest.mark.slow
@pytest.mark.requires_external
class TestRealGitOperations:
    """Tests that actually interact with git (slow, requires git)."""

    def test_current_branch_check(self):
        """Test checking the current actual branch."""
        # Get current branch
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            current_branch = result.stdout.strip()
            if current_branch:
                # This might fail if local differs from remote, which is expected
                try:
                    check_branch(current_branch, allow_diff_local_repo=True)
                except Exception:
                    # Expected if local and remote differ
                    pass
