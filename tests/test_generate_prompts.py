"""
Unit tests for prompt generation: size categories, grouping, fuzzy matching,
location cleaning, JSON extraction, and output format.
"""

import json
import pytest

from generate_prompts import (
    size_category,
    clean_location,
    group_lesions_for_case,
    extract_json_array,
    generate_template_prompts,
    SKIP_LOCATIONS,
)


# ============================================================
# size_category
# ============================================================

class TestSizeCategory:

    def test_tiny_boundary(self):
        assert size_category(0.004)[0] == "tiny"
        assert size_category(0.029)[0] == "tiny"

    def test_small_boundary(self):
        assert size_category(0.03)[0] == "small"
        assert size_category(0.179)[0] == "small"

    def test_medium_boundary(self):
        assert size_category(0.18)[0] == "medium"
        assert size_category(1.49)[0] == "medium"

    def test_large_boundary(self):
        assert size_category(1.5)[0] == "large"
        assert size_category(96.0)[0] == "large"

    def test_ordinals_ordered(self):
        _, o_tiny = size_category(0.001)
        _, o_small = size_category(0.05)
        _, o_medium = size_category(0.5)
        _, o_large = size_category(2.0)
        assert o_tiny < o_small < o_medium < o_large

    def test_very_small_value(self):
        name, ordinal = size_category(0.000119)
        assert name == "tiny"
        assert ordinal == 0


# ============================================================
# clean_location
# ============================================================

class TestCleanLocation:

    def test_ctx_lh_prefix(self):
        result = clean_location("ctx-lh-caudalmiddlefrontal")
        assert "left" in result
        assert "cortex" in result

    def test_ctx_rh_prefix(self):
        result = clean_location("ctx-rh-superiorfrontal")
        assert "right" in result
        assert "cortex" in result

    def test_left_cerebellum(self):
        result = clean_location("Left-Cerebellum-Exterior")
        assert "left" in result
        assert "cerebellar" in result.lower() or "cerebellum" in result.lower()

    def test_right_cerebellum_white_matter(self):
        result = clean_location("Right-Cerebellum-White-Matter")
        assert "right" in result
        assert "cerebellar" in result.lower() or "cerebellum" in result.lower()

    def test_brainstem(self):
        result = clean_location("Brain-Stem")
        assert "brainstem" in result.lower()

    def test_cbm_vermis(self):
        result = clean_location("Cbm_Vermis_VIII")
        assert "cerebellar" in result.lower() or "vermis" in result.lower()

    def test_ventricle(self):
        assert "third" in clean_location("3rd-Ventricle").lower()
        assert "fourth" in clean_location("4th-Ventricle").lower()

    def test_left_lateral_ventricle(self):
        result = clean_location("Left-Lateral-Ventricle")
        assert "left" in result
        assert "ventricle" in result.lower()

    def test_left_thalamus(self):
        result = clean_location("Left-Thalamus")
        assert "left" in result
        assert "thalamus" in result.lower()


# ============================================================
# group_lesions_for_case — fuzzy matching
# ============================================================

def _make_lesion(ln, size_ml, location, bbox_sag=5, bbox_cor=5, bbox_ax=5):
    return {
        "lesion_number": ln,
        "size_ml": size_ml,
        "location": location,
        "location_modifier": "in",
        "bbox_sag_mm": bbox_sag,
        "bbox_cor_mm": bbox_cor,
        "bbox_ax_mm": bbox_ax,
    }


class TestGrouping:

    def test_single_lesion_single_group(self):
        lesions = [_make_lesion(1, 0.5, "ctx-rh-precuneus", 15, 12, 18)]
        groups = group_lesions_for_case(lesions)

        assert len(groups["lesion_groups"]) == 1
        assert groups["lesion_groups"][0]["label_set"] == [1]
        assert len(groups["region_groups"]) == 1
        assert groups["global_group"]["all_lesion_numbers"] == [1]

    def test_two_lesions_same_region_same_category(self):
        """Two small lesions in same region → one lesion group."""
        lesions = [
            _make_lesion(1, 0.01, "ctx-rh-frontal", 6, 5, 7),
            _make_lesion(2, 0.02, "ctx-rh-frontal", 8, 9, 7),
        ]
        groups = group_lesions_for_case(lesions)

        assert len(groups["lesion_groups"]) == 1
        assert set(groups["lesion_groups"][0]["label_set"]) == {1, 2}

    def test_fuzzy_matching_includes_adjacent_categories(self):
        """Small prompt should include tiny+small+medium in same region."""
        lesions = [
            _make_lesion(1, 0.01,  "R", 3, 3, 3),    # tiny (ord=0)
            _make_lesion(2, 0.05,  "R", 6, 6, 6),     # small (ord=1)
            _make_lesion(3, 0.5,   "R", 12, 12, 12),   # medium (ord=2)
            _make_lesion(4, 2.0,   "R", 25, 25, 25),   # large (ord=3)
        ]
        groups = group_lesions_for_case(lesions)

        # Find the "small" group
        small_group = [g for g in groups["lesion_groups"]
                       if g["representative_category"] == "small"
                       or g["size_ord"] == 1]
        assert len(small_group) >= 1
        # Small (ord=1) should include tiny(0), small(1), medium(2) — NOT large(3)
        label_set = set(small_group[0]["label_set"])
        assert 1 in label_set  # tiny
        assert 2 in label_set  # small
        assert 3 in label_set  # medium
        assert 4 not in label_set  # large excluded (2 categories away)

    def test_fuzzy_matching_excludes_distant_categories(self):
        """Tiny and large are 3 apart → never in same fuzzy set."""
        lesions = [
            _make_lesion(1, 0.01, "R", 2, 2, 2),    # tiny (0)
            _make_lesion(2, 2.0,  "R", 25, 25, 25),   # large (3)
        ]
        groups = group_lesions_for_case(lesions)

        # Tiny group should NOT include large
        tiny_group = [g for g in groups["lesion_groups"] if g["size_ord"] == 0][0]
        assert 1 in tiny_group["label_set"]
        assert 2 not in tiny_group["label_set"]

        # Large group should NOT include tiny
        large_group = [g for g in groups["lesion_groups"] if g["size_ord"] == 3][0]
        assert 2 in large_group["label_set"]
        assert 1 not in large_group["label_set"]

    def test_deduplication_merges_identical_label_sets(self):
        """Two adjacent categories in same region with same fuzzy set → merge."""
        # Only tiny and small in region → both fuzzy-match to {1, 2}
        lesions = [
            _make_lesion(1, 0.01, "R", 3, 3, 3),   # tiny
            _make_lesion(2, 0.05, "R", 6, 6, 6),    # small
        ]
        groups = group_lesions_for_case(lesions)

        # Should deduplicate to 1 lesion group (both fuzzy to {1, 2})
        assert len(groups["lesion_groups"]) == 1
        assert set(groups["lesion_groups"][0]["label_set"]) == {1, 2}

    def test_different_regions_separate_groups(self):
        """Same category but different regions → separate groups."""
        lesions = [
            _make_lesion(1, 0.01, "ctx-lh-frontal"),
            _make_lesion(2, 0.01, "ctx-rh-frontal"),
        ]
        groups = group_lesions_for_case(lesions)

        assert len(groups["lesion_groups"]) == 2
        assert groups["lesion_groups"][0]["label_set"] == [1]
        assert groups["lesion_groups"][1]["label_set"] == [2]

    def test_region_groups_include_all_lesions(self):
        """Region group includes ALL lesions regardless of size."""
        lesions = [
            _make_lesion(1, 0.01, "R"),   # tiny
            _make_lesion(2, 2.0,  "R"),    # large
        ]
        groups = group_lesions_for_case(lesions)

        assert set(groups["region_groups"]["R"]["lesion_numbers"]) == {1, 2}

    def test_global_group_includes_everything(self):
        lesions = [
            _make_lesion(1, 0.01, "A"),
            _make_lesion(2, 0.5,  "B"),
            _make_lesion(3, 0.1,  "C"),
        ]
        groups = group_lesions_for_case(lesions)

        assert groups["global_group"]["all_lesion_numbers"] == [1, 2, 3]
        assert groups["global_group"]["lesion_count"] == 3

    def test_skip_locations_excluded_from_lesion_and_region_groups(self):
        """Lesions in Unknown/CSF locations should be skipped for lesion/region groups."""
        lesions = [
            _make_lesion(1, 0.01, "Unknown"),
            _make_lesion(2, 0.01, "CSF"),
            _make_lesion(3, 0.01, "ctx-lh-frontal"),
        ]
        groups = group_lesions_for_case(lesions)

        # Only ctx-lh-frontal should have lesion/region groups
        assert len(groups["lesion_groups"]) == 1
        assert len(groups["region_groups"]) == 1
        # But global includes everything
        assert groups["global_group"]["all_lesion_numbers"] == [1, 2, 3]


# ============================================================
# extract_json_array
# ============================================================

class TestExtractJsonArray:

    def test_clean_json(self):
        text = '[{"prompt": "hello"}, {"prompt": "world"}]'
        result = extract_json_array(text)
        assert len(result) == 2

    def test_markdown_fences(self):
        text = '```json\n[{"prompt": "test"}]\n```'
        result = extract_json_array(text)
        assert result is not None
        assert result[0]["prompt"] == "test"

    def test_trailing_text(self):
        text = 'Here are the prompts:\n[{"prompt": "a"}]\nI hope these help!'
        result = extract_json_array(text)
        assert result is not None
        assert len(result) == 1

    def test_invalid_json_returns_none(self):
        assert extract_json_array("not json at all") is None
        assert extract_json_array("{not an array}") is None

    def test_empty_array(self):
        result = extract_json_array("[]")
        assert result == []

    def test_nested_brackets_in_strings(self):
        text = '[{"prompt": "lesion [1] in brain"}]'
        result = extract_json_array(text)
        assert result is not None
        assert "lesion [1]" in result[0]["prompt"]


# ============================================================
# Template mode output format
# ============================================================

class TestTemplateOutput:

    def test_output_has_required_keys(self):
        lesions = [_make_lesion(1, 0.5, "ctx-rh-frontal", 20, 18, 15)]
        prompts = generate_template_prompts(lesions)

        for p in prompts:
            assert "prompt" in p
            assert "lesion_numbers" in p
            assert "prompt_type" in p
            assert isinstance(p["prompt"], str)
            assert isinstance(p["lesion_numbers"], list)
            assert p["prompt_type"] in ("lesion", "region", "global")

    def test_all_three_prompt_types_present(self):
        lesions = [_make_lesion(1, 0.5, "ctx-rh-frontal", 20, 18, 15)]
        prompts = generate_template_prompts(lesions)

        types = {p["prompt_type"] for p in prompts}
        assert types == {"lesion", "region", "global"}

    def test_lesion_numbers_are_valid_ids(self):
        lesions = [
            _make_lesion(1, 0.01, "R"),
            _make_lesion(2, 0.5, "R"),
        ]
        prompts = generate_template_prompts(lesions)

        valid_ids = {1, 2}
        for p in prompts:
            for ln in p["lesion_numbers"]:
                assert ln in valid_ids

    def test_global_prompts_include_all_lesions(self):
        lesions = [
            _make_lesion(1, 0.01, "A"),
            _make_lesion(2, 0.5, "B"),
        ]
        prompts = generate_template_prompts(lesions)

        global_prompts = [p for p in prompts if p["prompt_type"] == "global"]
        for p in global_prompts:
            assert set(p["lesion_numbers"]) == {1, 2}


# ============================================================
# CSV loading
# ============================================================

from generate_prompts import load_csv_metadata


class TestCSVLoading:

    def test_load_csv_metadata_basic(self, tmp_path):
        """Basic CSV with well-formed rows should load correctly."""
        csv_path = tmp_path / "case.csv"
        csv_path.write_text(
            "lesion_number,size_ml,location,location_modifier,axial_slice,bbox_sag,bbox_cor,bbox_ax\n"
            "1,0.05,ctx-lh-frontal,in,45,10.0,8.0,6.0\n"
            "2,0.2,Right-Putamen,near,60,15.0,12.0,9.0\n"
        )
        lesions = load_csv_metadata(csv_path)
        assert len(lesions) == 2
        assert lesions[0]["lesion_number"] == 1
        assert lesions[0]["size_ml"] == pytest.approx(0.05)
        assert lesions[0]["location"] == "ctx-lh-frontal"
        assert lesions[0]["location_modifier"] == "in"
        assert lesions[1]["lesion_number"] == 2
        assert "bbox_sag_mm" in lesions[1]
        assert lesions[1]["bbox_sag_mm"] == pytest.approx(15.0)

    def test_load_csv_metadata_malformed_rows(self, tmp_path):
        """Rows with too few columns or invalid values should be skipped."""
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text(
            "lesion_number,size_ml,location,location_modifier\n"
            "1,0.05,ctx-lh-frontal,in\n"
            "bad,row\n"
            "3,not_a_number,ctx-rh-frontal,in\n"
            "4,0.1,Right-Putamen,near\n"
        )
        lesions = load_csv_metadata(csv_path)
        # Only rows 1 and 4 should load (row 2 too short, row 3 invalid float)
        assert len(lesions) == 2
        assert lesions[0]["lesion_number"] == 1
        assert lesions[1]["lesion_number"] == 4

    def test_load_csv_metadata_empty(self, tmp_path):
        """CSV with only header should return empty list."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("lesion_number,size_ml,location,location_modifier\n")
        lesions = load_csv_metadata(csv_path)
        assert lesions == []


# ============================================================
# FreeSurfer location cleaning (additional tests)
# ============================================================

class TestCleanLocationFreeSurfer:

    def test_cortical_concatenated_names(self):
        """FreeSurfer concatenated cortical labels should expand correctly."""
        assert "inferior parietal" in clean_location("ctx-lh-inferiorparietal")
        assert "caudal middle frontal" in clean_location("ctx-rh-caudalmiddlefrontal")
        assert "superior frontal" in clean_location("ctx-lh-superiorfrontal")
        assert "lateral occipital" in clean_location("ctx-rh-lateraloccipital")
        assert "pars triangularis" in clean_location("ctx-lh-parstriangularis")

    def test_cortical_has_hemisphere(self):
        """Cortical labels should include hemisphere (left/right)."""
        assert "left" in clean_location("ctx-lh-inferiorparietal")
        assert "right" in clean_location("ctx-rh-superiorfrontal")

    def test_cortical_has_cortex(self):
        """Cortical labels should include 'cortex' suffix."""
        assert "cortex" in clean_location("ctx-lh-precuneus")
        assert "cortex" in clean_location("ctx-rh-insula")

    def test_subcortical_lowercase(self):
        """Subcortical labels should be lowercased."""
        assert clean_location("Right-Putamen") == "right putamen"

    def test_subcortical_left(self):
        result = clean_location("Left-Thalamus")
        assert "left" in result
        assert "thalamus" in result.lower()


# ============================================================
# Tiny lesion filtering in template mode
# ============================================================

class TestTinyLesionFilter:

    def test_tiny_lesions_skipped_in_template_mode(self):
        """All lesions < 0.002ml should produce no lesion-level prompts."""
        lesions = [
            _make_lesion(1, 0.001, "ctx-lh-frontal", 3, 3, 3),
            _make_lesion(2, 0.0015, "ctx-lh-frontal", 2, 2, 2),
        ]
        prompts = generate_template_prompts(lesions)

        lesion_prompts = [p for p in prompts if p["prompt_type"] == "lesion"]
        # All core lesions are < 0.002ml, so lesion-level prompts should be skipped
        assert len(lesion_prompts) == 0

        # But region and global prompts should still exist
        region_prompts = [p for p in prompts if p["prompt_type"] == "region"]
        global_prompts = [p for p in prompts if p["prompt_type"] == "global"]
        assert len(region_prompts) > 0
        assert len(global_prompts) > 0

    def test_mixed_sizes_only_filters_tiny_groups(self):
        """Groups with at least one >= 0.002ml lesion should still get prompts."""
        lesions = [
            _make_lesion(1, 0.001, "ctx-lh-frontal", 3, 3, 3),   # tiny, < 0.002
            _make_lesion(2, 0.01, "ctx-rh-frontal", 6, 6, 6),     # small, >= 0.002
        ]
        prompts = generate_template_prompts(lesions)

        lesion_prompts = [p for p in prompts if p["prompt_type"] == "lesion"]
        # At least the small lesion's group should produce lesion-level prompts
        assert len(lesion_prompts) > 0
        # The small lesion group should include lesion 2
        all_lesion_nums = set()
        for p in lesion_prompts:
            all_lesion_nums.update(p["lesion_numbers"])
        assert 2 in all_lesion_nums
