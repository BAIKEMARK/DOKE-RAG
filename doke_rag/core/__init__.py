"""
DOKE-RAG Core Framework

This module is based on LightRAG (https://github.com/HKUDS/LightRAG)
Copyright (c) 2024 HKUDS
Modified by DOKE-RAG Team for domain-oriented knowledge enhancement

The original LightRAG project is licensed under the MIT License.
See LICENSE.LightRAG file in this directory for details.
"""

# Copyright (c) 2024 HKUDS
# Copyright (c) 2025 DOKE-RAG Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This software is based on LightRAG (https://github.com/HKUDS/LightRAG)
# Original LightRAG Copyright (c) 2024 HKUDS
# Modified by DOKE-RAG Team
# Modification date: 2025-11

from .lightrag import LightRAG as LightRAG
from .lightrag import QueryParam as QueryParam

__version__ = "1.2.4-doke"  # DOKE-RAG version
__api_version__ = __version__  # API version for User-Agent headers
__author__ = "HKUDS, DOKE-RAG Team"
__url__ = "https://github.com/HKUDS/LightRAG"
