"""Constants and fixed configurations for MS²PIP."""

# Models and their properties
# id is passed to get_predictions to select model
# ion_types is required to write the ion types in the headers of the result files
# features_version is required to select the features version
MODELS = {
    "CID": {
        "id": 0,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20190107_CID_train_B.xgboost",
            "y": "model_20190107_CID_train_Y.xgboost",
        },
        "model_hash": {
            "model_20190107_CID_train_B.xgboost": "4398c6ebe23e2f37c0aca42b095053ecea6fb427",
            "model_20190107_CID_train_Y.xgboost": "e0a9eb37e50da35a949d75807d66fb57e44aca0f",
        },
    },
    "HCD2019": {
        "id": 1,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
    },
    "TTOF5600": {
        "id": 2,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20190107_TTOF5600_train_B.xgboost",
            "y": "model_20190107_TTOF5600_train_Y.xgboost",
        },
        "model_hash": {
            "model_20190107_TTOF5600_train_B.xgboost": "ab2e28dfbc4ee60640253b0b4c127fc272c9d0ed",
            "model_20190107_TTOF5600_train_Y.xgboost": "f8e9ddd8ca78ace06f67460a2fea0d8fa2623452",
        },
    },
    "TMT": {
        "id": 3,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
    },
    "iTRAQ": {
        "id": 4,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20190107_iTRAQ_train_B.xgboost",
            "y": "model_20190107_iTRAQ_train_Y.xgboost",
        },
        "model_hash": {
            "model_20190107_iTRAQ_train_B.xgboost": "b8d94ad329a245210c652a5b35d724d2c74d0d50",
            "model_20190107_iTRAQ_train_Y.xgboost": "56ae87d56fd434b53fcc1d291745cabb7baf463a",
        },
    },
    "iTRAQphospho": {
        "id": 5,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20190107_iTRAQphospho_train_B.xgboost",
            "y": "model_20190107_iTRAQphospho_train_Y.xgboost",
        },
        "model_hash": {
            "model_20190107_iTRAQphospho_train_B.xgboost": "e283b158cc50e219f42f93be624d0d0ac01d6b49",
            "model_20190107_iTRAQphospho_train_Y.xgboost": "261b2e1810a299ed7ebf193ce1fb81a608c07d3b",
        },
    },
    # ETD': {'id': 6, 'ion_types': ['B', 'Y', 'C', 'Z'], 'peaks_version': 'etd', 'features_version': 'normal'},
    "HCDch2": {
        "id": 7,
        "ion_types": ["B", "Y", "B2", "Y2"],
        "peaks_version": "ch2",
        "features_version": "normal",
    },
    "CIDch2": {
        "id": 8,
        "ion_types": ["B", "Y", "B2", "Y2"],
        "peaks_version": "ch2",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20190107_CID_train_B.xgboost",
            "y": "model_20190107_CID_train_Y.xgboost",
            "b2": "model_20190107_CID_train_B2.xgboost",
            "y2": "model_20190107_CID_train_Y2.xgboost",
        },
        "model_hash": {
            "model_20190107_CID_train_B.xgboost": "4398c6ebe23e2f37c0aca42b095053ecea6fb427",
            "model_20190107_CID_train_Y.xgboost": "e0a9eb37e50da35a949d75807d66fb57e44aca0f",
            "model_20190107_CID_train_B2.xgboost": "602f2fc648890aebbbe2646252ade658af3221a3",
            "model_20190107_CID_train_Y2.xgboost": "4e4ad0f1d4606c17015aae0f74edba69f684d399",
        },
    },
    "HCD2021": {
        "id": 9,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20210416_HCD2021_B.xgboost",
            "y": "model_20210416_HCD2021_Y.xgboost",
        },
        "model_hash": {
            "model_20210416_HCD2021_B.xgboost": "c086c599f618b199bbb36e2411701fb2866b24c8",
            "model_20210416_HCD2021_Y.xgboost": "22a5a137e29e69fa6d4320ed7d701b61cbdc4fcf",
        },
    },
    "Immuno-HCD": {
        "id": 10,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20210316_Immuno_HCD_B.xgboost",
            "y": "model_20210316_Immuno_HCD_Y.xgboost",
        },
        "model_hash": {
            "model_20210316_Immuno_HCD_B.xgboost": "977466d378de2e89c6ae15b4de8f07800d17a7b7",
            "model_20210316_Immuno_HCD_Y.xgboost": "71948e1b9d6c69cb69b9baf84d361a9f80986fea",
        },
    },
    "CID-TMT": {
        "id": 11,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20220104_CID_TMT_B.xgboost",
            "y": "model_20220104_CID_TMT_Y.xgboost",
        },
        "model_hash": {
            "model_20220104_CID_TMT_B.xgboost": "fa834162761a6ae444bb6523c9c1a174b9738389",
            "model_20220104_CID_TMT_Y.xgboost": "299539179ca55d4ac82e9aed6a4e0bd134a9a41e",
        },
    },
    "timsTOF2023": {
        "id": 12,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20230912_timsTOF_B.xgboost",
            "y": "model_20230912_timsTOF_Y.xgboost",
        },
        "model_hash": {
            "model_20230912_timsTOF_B.xgboost": "6beb557052455310d8c66311c86866dda8291f4b",
            "model_20230912_timsTOF_Y.xgboost": "8edd87e0fba5f338d0a0881b5afbcf2f48ec5268",
        },
    },
    "timsTOF2024": {
        "id": 13,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20240105_timsTOF_B.xgboost",
            "y": "model_20240105_timsTOF_Y.xgboost",
        },
        "model_hash": {
            "model_20240105_timsTOF_B.xgboost": "d70e145c15cf2bfa30968077a68409699b2fa541",
            "model_20240105_timsTOF_Y.xgboost": "3f0414ee1ad7cff739e0d6242e25bfc22b6ebfe5",
        },
    },
}


MODELS["HCD"] = MODELS["HCD2021"]
MODELS["timsTOF"] = MODELS["timsTOF2024"]
