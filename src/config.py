# -*- coding: utf-8 -*-
"""실행 경로 설정.

데이터 위치는 환경변수 HOUSING_DATA_DIR 로 지정한다.
지정하지 않으면 저장소 안의 data/ 를 본다.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

DATA_DIR = os.environ.get("HOUSING_DATA_DIR", os.path.join(_ROOT, "data"))
RESULTS_DIR = os.environ.get("HOUSING_RESULTS_DIR", os.path.join(_ROOT, "results"))

os.makedirs(RESULTS_DIR, exist_ok=True)