"""
Vertical Domain Data Pipeline
Medical and Legal domain data collection, cleaning, and preprocessing
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PHIPattern:
    """PHI detection pattern configuration"""

    name: str
    pattern: str
    category: str
    replacement: str = "[REDACTED]"


class MedicalPHIRemover:
    """HIPAA-compliant PHI removal for medical data"""

    PHI_PATTERNS = [
        PHIPattern("name", r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", "person"),
        PHIPattern("phone", r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "contact"),
        PHIPattern(
            "email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "contact"
        ),
        PHIPattern("ssn", r"\b\d{3}-\d{2}-\d{4}\b", "id"),
        PHIPattern("mrn", r"\bMRN[:\s]*\d+\b", "id"),
        PHIPattern("dob", r"\b\d{1,2}/\d{1,2}/\d{4}\b", "date"),
        PHIPattern(
            "address",
            r"\b\d+\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b",
            "location",
        ),
        PHIPattern("ip_address", r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "id"),
    ]

    def __init__(self):
        self.compiled_patterns = [
            (re.compile(p.pattern, re.IGNORECASE), p) for p in self.PHI_PATTERNS
        ]

    def remove_phi(self, text: str) -> Tuple[str, List[str]]:
        """Remove PHI from text and return redaction log"""
        redactions = []

        for pattern, phi_info in self.compiled_patterns:
            matches = pattern.findall(text)
            for match in matches:
                text = text.replace(match, phi_info.replacement)
                redactions.append(
                    {
                        "type": phi_info.name,
                        "category": phi_info.category,
                        "value": match[:20] + "..." if len(match) > 20 else match,
                        "replacement": phi_info.replacement,
                    }
                )

        return text, redactions

    def batch_process(self, texts: List[str]) -> List[str]:
        """Process multiple texts"""
        return [self.remove_phi(t)[0] for t in texts]


class LegalPHIRemover:
    """Personal information removal for legal documents"""

    LEGAL_PATTERNS = [
        PHIPattern(
            "case_number", r"\b\d{4}[年沪深]?\d{2,6}[刑民行商执]?第?\d+\b", "case"
        ),
        PHIPattern(
            "name",
            r"\b[李王张刘陈杨黄赵吴周徐孙马朱胡郭何高林郑]\s+[李王张刘陈杨黄赵吴周徐孙马朱胡郭何高林郑]\b",
            "person",
        ),
        PHIPattern("phone", r"\b1[3-9]\d{9}\b", "contact"),
        PHIPattern("id_number", r"\b\d{17}[\dXx]\b", "id"),
        PHIPattern(
            "address", r"\b[^\s]{2,10}省[^\s]{2,10}市[^\s]{2,10}(区|县)\b", "location"
        ),
    ]

    def __init__(self):
        self.compiled_patterns = [
            (re.compile(p.pattern), p) for p in self.LEGAL_PATTERNS
        ]

    def remove_phi(self, text: str) -> Tuple[str, List[str]]:
        """Remove PHI from legal text"""
        redactions = []

        for pattern, phi_info in self.compiled_patterns:
            matches = pattern.findall(text)
            for match in matches:
                text = text.replace(match, phi_info.replacement)
                redactions.append(
                    {
                        "type": phi_info.name,
                        "category": phi_info.category,
                        "value": match,
                        "replacement": phi_info.replacement,
                    }
                )

        return text, redactions

    def batch_process(self, texts: List[str]) -> List[str]:
        """Process multiple legal documents"""
        return [self.remove_phi(t)[0] for t in texts]


class MedicalTextProcessor:
    """Medical domain text processing"""

    MEDICAL_TERMS = {
        "ecmo": "extracorporeal membrane oxygenation",
        "bp": "blood pressure",
        "ct": "computed tomography",
        "mri": "magnetic resonance imaging",
        "icu": "intensive care unit",
        "cpr": "cardiopulmonary resuscitation",
        "ekg": "electrocardiogram",
        "cabg": "coronary artery bypass grafting",
    }

    def __init__(self):
        self.abbreviation_pattern = re.compile(
            r"\b(" + "|".join(self.MEDICAL_TERMS.keys()) + r")\b", re.IGNORECASE
        )

    def expand_abbreviations(self, text: str) -> str:
        """Expand medical abbreviations to full terms"""

        def replace(match):
            term = match.group(1).lower()
            return self.MEDICAL_TERMS.get(term, match.group(0))

        return self.abbreviation_pattern.sub(replace, text)

    def extract_medical_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract medical entities (symptoms, diagnoses, medications)"""
        entities = []

        diagnosis_pattern = (
            r"(?:diagnosed with|diagnosis|diagnosed as)\s+([A-Z][a-z]+(?:\s+[a-z]+)?)"
        )
        medication_pattern = (
            r"(?:prescribed|medication|drug)\s+([A-Z][a-z]+(?:\s+[a-z]+)?)"
        )

        for match in re.finditer(diagnosis_pattern, text):
            entities.append(
                {
                    "text": match.group(1),
                    "type": "diagnosis",
                    "position": (match.start(), match.end()),
                }
            )

        for match in re.finditer(medication_pattern, text):
            entities.append(
                {
                    "text": match.group(1),
                    "type": "medication",
                    "position": (match.start(), match.end()),
                }
            )

        return entities

    def normalize_measurements(self, text: str) -> str:
        """Normalize medical measurements to standard format"""
        text = re.sub(r"(\d+)\s*mg/kg", r"\1 mg/kg", text)
        text = re.sub(r"(\d+)\s*mmHg", r"\1 mmHg", text)
        text = re.sub(r"(\d+\.?\d*)\s*(mg|mcg|g|ml)", r"\1 \2", text)
        return text


class LegalTextProcessor:
    """Legal domain text processing"""

    LEGAL_TERMS = {
        "plaintiff": "民事原告",
        "defendant": "被告",
        "appellant": "上诉人",
        "respondent": "被上诉人",
        "plaintiff": "原告",
        "jurisdiction": "管辖权",
        "precedent": "先例",
        "statute": "法规",
        "liability": "责任",
    }

    def __init__(self):
        self.article_pattern = re.compile(r"第\s*\d+\s*条")
        self.section_pattern = re.compile(r"(?:第|条|款|项)\s*\d+")

    def extract_citations(self, text: str) -> List[Dict[str, str]]:
        """Extract legal citations from text"""
        citations = []

        law_pattern = r"《([^》]+)》第(\d+)条"
        for match in re.finditer(law_pattern, text):
            citations.append(
                {
                    "law": match.group(1),
                    "article": match.group(2),
                    "full_text": match.group(0),
                    "position": (match.start(), match.end()),
                }
            )

        return citations

    def extract_legal_issues(self, text: str) -> List[str]:
        """Identify legal issues in text"""
        issues = []

        issue_keywords = [
            "合同纠纷",
            "侵权责任",
            "财产纠纷",
            "劳动争议",
            "知识产权",
            "婚姻家庭",
            "交通事故",
            "医疗损害",
        ]

        for keyword in issue_keywords:
            if keyword in text:
                issues.append(keyword)

        return issues

    def normalize_case_names(self, text: str) -> str:
        """Normalize case name format"""
        text = re.sub(r"(\d{4})年第(\d+)号", r"\1>\2>", text)
        return text


@dataclass
class DomainDataset:
    """Container for domain-specific dataset"""

    name: str
    instructions: List[str]
    responses: List[str]
    metadata: Dict[str, Any]

    def __len__(self):
        return len(self.instructions)

    def save(self, path: str):
        """Save dataset to JSON"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": self.name,
                    "instructions": self.instructions,
                    "responses": self.responses,
                    "metadata": self.metadata,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "DomainDataset":
        """Load dataset from JSON"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            name=data["name"],
            instructions=data["instructions"],
            responses=data["responses"],
            metadata=data.get("metadata", {}),
        )


class DataQualityFilter:
    """Filter low-quality samples from dataset"""

    def __init__(self, min_length: int = 50, max_length: int = 4096):
        self.min_length = min_length
        self.max_length = max_length

    def check_length(self, text: str) -> bool:
        """Check if text length is within acceptable range"""
        return self.min_length <= len(text) <= self.max_length

    def check_completeness(self, text: str) -> bool:
        """Check if text is complete (ends properly)"""
        incomplete_indicators = ["...", "...", "continue", "continued"]
        return not any(text.rstrip().endswith(ind) for ind in incomplete_indicators)

    def check_duplicates(self, texts: List[str]) -> List[int]:
        """Find duplicate indices"""
        seen = {}
        duplicate_indices = []

        for i, text in enumerate(texts):
            normalized = text.lower().strip()[:100]
            if normalized in seen:
                duplicate_indices.append(i)
            else:
                seen[normalized] = i

        return duplicate_indices

    def filter(
        self, instructions: List[str], responses: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Filter dataset based on quality criteria"""
        valid_indices = []

        for i in range(len(instructions)):
            if not self.check_length(responses[i]):
                continue
            if not self.check_completeness(responses[i]):
                continue
            valid_indices.append(i)

        return (
            [instructions[i] for i in valid_indices],
            [responses[i] for i in valid_indices],
        )


class MedicalDataPipeline:
    """Complete data pipeline for medical domain"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.phi_remover = MedicalPHIRemover()
        self.text_processor = MedicalTextProcessor()
        self.quality_filter = DataQualityFilter(min_length=100)

        self.processing_stats = {
            "total_samples": 0,
            "phi_removed": 0,
            "filtered": 0,
            "success": 0,
        }

    def load_medical_corpus(self, paths: List[str]) -> List[Dict]:
        """Load medical corpus from multiple sources"""
        documents = []

        for path in paths:
            if path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    documents.extend(data if isinstance(data, list) else [data])
            elif path.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    documents.append({"text": f.read(), "source": path})

        logger.info(f"Loaded {len(documents)} documents from {len(paths)} sources")
        return documents

    def build_qa_pairs(self, documents: List[Dict]) -> Tuple[List[str], List[str]]:
        """Build instruction-response pairs from medical documents"""
        instructions = []
        responses = []

        template_patterns = [
            (
                "Based on the following clinical note, what is the likely diagnosis?\n\n{text}",
                "Based on the clinical presentation, the likely diagnosis is...",
            ),
            (
                "What treatment would you recommend for this case?\n\n{text}",
                "The recommended treatment approach includes...",
            ),
            (
                "Extract the key clinical findings from this note:\n\n{text}",
                "The key clinical findings are...",
            ),
        ]

        for doc in tqdm(documents, desc="Building QA pairs"):
            text = doc.get("text", "")

            if len(text) < 200:
                continue

            cleaned_text, _ = self.phi_remover.remove_phi(text)

            for instr_template, resp_template in template_patterns:
                instruction = instr_template.format(text=cleaned_text[:1000])
                response = resp_template.format(text=cleaned_text[:1000])

                instructions.append(instruction)
                responses.append(response)

        self.processing_stats["total_samples"] = len(instructions)
        logger.info(f"Built {len(instructions)} QA pairs")

        return instructions, responses

    def process(self, input_paths: List[str]) -> DomainDataset:
        """Execute full pipeline"""
        documents = self.load_medical_corpus(input_paths)

        instructions, responses = self.build_qa_pairs(documents)

        filtered_inst, filtered_resp = self.quality_filter.filter(
            instructions, responses
        )

        self.processing_stats["filtered"] = len(instructions) - len(filtered_inst)
        self.processing_stats["success"] = len(filtered_inst)

        dataset = DomainDataset(
            name="medical_sft_dataset",
            instructions=filtered_inst,
            responses=filtered_resp,
            metadata={
                "domain": "medical",
                "created_at": datetime.now().isoformat(),
                "stats": self.processing_stats,
                "pipeline_version": "1.0",
            },
        )

        output_path = self.output_dir / "medical_dataset.json"
        dataset.save(str(output_path))
        logger.info(f"Dataset saved to {output_path}")

        return dataset


class LegalDataPipeline:
    """Complete data pipeline for legal domain"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.phi_remover = LegalPHIRemover()
        self.text_processor = LegalTextProcessor()
        self.quality_filter = DataQualityFilter(min_length=150)

        self.processing_stats = {
            "total_samples": 0,
            "phi_removed": 0,
            "citations_found": 0,
            "filtered": 0,
            "success": 0,
        }

    def load_legal_corpus(self, paths: List[str]) -> List[Dict]:
        """Load legal documents from multiple sources"""
        documents = []

        for path in paths:
            if path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    documents.extend(data if isinstance(data, list) else [data])
            elif path.endswith(".txt") or path.endswith(".pdf"):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    documents.append({"text": f.read(), "source": path})

        logger.info(f"Loaded {len(documents)} legal documents")
        return documents

    def extract_judgment_info(self, text: str) -> Dict[str, Any]:
        """Extract key information from judgment documents"""
        info = {
            "case_type": None,
            "legal_issues": [],
            "citations": [],
            "judgment": None,
        }

        case_types = ["民事", "刑事", "行政", "商事", "劳动"]
        for ct in case_types:
            if ct in text:
                info["case_type"] = ct
                break

        info["legal_issues"] = self.text_processor.extract_legal_issues(text)
        info["citations"] = self.text_processor.extract_citations(text)

        judgment_patterns = [
            r"判决如下[：:](.*?)(?=本判决|$)",
            r"裁定如下[：:](.*?)(?=本裁定|$)",
        ]

        for pattern in judgment_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                info["judgment"] = match.group(1).strip()
                break

        return info

    def build_legal_qa_pairs(
        self, documents: List[Dict]
    ) -> Tuple[List[str], List[str]]:
        """Build instruction-response pairs from legal documents"""
        instructions = []
        responses = []

        templates = [
            (
                "根据以下案例摘要，分析其法律问题并引用相关法条：\n\n{text}",
                "该案例涉及的法律问题包括...",
            ),
            (
                "请分析以下判决书的事实认定和法律适用：\n\n{text}",
                "事实认定方面...法律适用方面...",
            ),
            (
                "基于以下案例，判断当事人可能面临的法律责任：\n\n{text}",
                "根据相关法律规定...",
            ),
        ]

        for doc in tqdm(documents, desc="Building legal QA pairs"):
            text = doc.get("text", "")

            if len(text) < 300:
                continue

            cleaned_text, _ = self.phi_remover.remove_phi(text)

            extracted_info = self.extract_judgment_info(cleaned_text)
            self.processing_stats["citations_found"] += len(extracted_info["citations"])

            for instr_template, resp_template in templates:
                instruction = instr_template.format(text=cleaned_text[:1500])
                response = resp_template.format(text=cleaned_text[:1500])

                instructions.append(instruction)
                responses.append(response)

        self.processing_stats["total_samples"] = len(instructions)

        return instructions, responses

    def process(self, input_paths: List[str]) -> DomainDataset:
        """Execute full pipeline"""
        documents = self.load_legal_corpus(input_paths)

        instructions, responses = self.build_legal_qa_pairs(documents)

        filtered_inst, filtered_resp = self.quality_filter.filter(
            instructions, responses
        )

        self.processing_stats["filtered"] = len(instructions) - len(filtered_inst)
        self.processing_stats["success"] = len(filtered_inst)

        dataset = DomainDataset(
            name="legal_sft_dataset",
            instructions=filtered_inst,
            responses=filtered_resp,
            metadata={
                "domain": "legal",
                "created_at": datetime.now().isoformat(),
                "stats": self.processing_stats,
                "pipeline_version": "1.0",
            },
        )

        output_path = self.output_dir / "legal_dataset.json"
        dataset.save(str(output_path))
        logger.info(f"Legal dataset saved to {output_path}")

        return dataset


class InstructionDataset(Dataset):
    """PyTorch Dataset for instruction fine-tuning"""

    def __init__(
        self,
        instructions: List[str],
        responses: List[str],
        tokenizer,
        max_length: int = 2048,
    ):
        self.instructions = instructions
        self.responses = responses
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        instruction = self.instructions[idx]
        response = self.responses[idx]

        prompt = f"Instruction: {instruction}\nResponse: {response}"

        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


def main():
    """Example usage of data pipelines"""
    import argparse

    parser = argparse.ArgumentParser(description="Vertical Domain Data Pipeline")
    parser.add_argument("--domain", choices=["medical", "legal"], required=True)
    parser.add_argument("--input", nargs="+", required=True, help="Input file paths")
    parser.add_argument("--output", default="./output", help="Output directory")

    args = parser.parse_args()

    if args.domain == "medical":
        pipeline = MedicalDataPipeline(args.output)
        dataset = pipeline.process(args.input)
    else:
        pipeline = LegalDataPipeline(args.output)
        dataset = pipeline.process(args.input)

    print(f"\nProcessing complete:")
    print(f"  Total samples: {dataset.metadata['stats']['total_samples']}")
    print(f"  Filtered: {dataset.metadata['stats']['filtered']}")
    print(f"  Final dataset size: {len(dataset)}")


if __name__ == "__main__":
    main()
