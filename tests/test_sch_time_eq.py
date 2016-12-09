#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_sch_time_eq
----------------------------------

Tests for `sch_time_eq` module.
"""


import sys
import unittest
from contextlib import contextmanager
from click.testing import CliRunner

from sch_time_eq import sch_time_eq
from sch_time_eq import cli



class TestSch_time_eq(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_000_something(self):
        pass

    def test_command_line_interface(self):
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'sch_time_eq.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output