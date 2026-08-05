import os
import cv2
import uuid
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Union

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Configure logging
logger = logging.getLogger("ChallanGenerator")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [ChallanGenerator] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

@dataclass
class ChallanPayload:
    """Represents the structured return payload for a generated challan."""
    challan_id: str
    timestamp: str
    owner_name: str
    owner_phone: str
    license_plate: str
    violation_type: str  # Comma-separated or consolidated string
    fine_amount: int
    evidence_image_path: str
    pdf_path: str
    clip_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataclass instance to a clean Python dictionary."""
        return {
            "challan_id": self.challan_id,
            "timestamp": self.timestamp,
            "owner_name": self.owner_name,
            "owner_phone": self.owner_phone,
            "license_plate": self.license_plate,
            "violation_type": self.violation_type,
            "fine_amount": self.fine_amount,
            "evidence_image_path": self.evidence_image_path,
            "pdf_path": self.pdf_path,
            "clip_path": self.clip_path
        }

class ChallanGenerator:
    def __init__(self, assets_dir: Optional[str] = None, challans_dir: Optional[str] = None, outputs_dir: Optional[str] = None):
        """
        Initializes the ChallanGenerator.
        """
        base_dir = Path(__file__).parent.parent.resolve()
        self.assets_dir = Path(assets_dir) if assets_dir else base_dir / "assets" / "detected_plates"
        self.challans_dir = Path(challans_dir) if challans_dir else base_dir / "generated_challans"
        self.outputs_dir = Path(outputs_dir) if outputs_dir else base_dir / "outputs"
        
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.challans_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Penalties Mapping
        self.fine_mapping = {
            "NO HELMET": 1000,
            "TRIPLE RIDING": 1000,
            "OVERSPEEDING": 2000,
            "WRONG WAY": 2000,
            "ZEBRA CROSSING OBSTRUCTION": 500,
            "ZEBRA OBSTRUCTION": 500  # Defensive alias
        }

        # Mock VAHAN Database for matching known plates
        self.vahan_mock_db = {
            "RJ14AB1234": {"owner_name": "Niyati Kumawat", "owner_phone": "+91 98765 43210"},
            "MH12CD5678": {"owner_name": "Ramesh Kumar", "owner_phone": "+91 99999 88888"},
            "DL3CA5555": {"owner_name": "Priya Sharma", "owner_phone": "+91 98111 22222"},
            "KA03MM9999": {"owner_name": "Anil Kumble", "owner_phone": "+91 94444 55555"}
        }

    def get_vehicle_owner_info(self, plate_number: Optional[str]) -> Dict[str, str]:
        """
        Queries the VAHAN mock database to return owner information.
        If plate_number is missing or unknown, generates a deterministic mock record.

        Args:
            plate_number (str): The vehicle's license plate text.

        Returns:
            Dict[str, str]: Dictionary containing 'owner_name' and 'owner_phone'.
        """
        if not plate_number:
            return {
                "owner_name": "Unknown Vehicle Owner",
                "owner_phone": "+91 00000 00000"
            }

        # Clean plate text (alphanumeric uppercase)
        clean_plate = "".join(e for e in plate_number if e.isalnum()).upper().strip()

        if clean_plate in self.vahan_mock_db:
            return self.vahan_mock_db[clean_plate]

        # Deterministic generation for unknown plates based on the plate characters
        hash_val = sum(ord(char) for char in clean_plate)
        
        first_names = ["Rajesh", "Vikram", "Sanjay", "Karan", "Sunita", "Anjali", "Neha", "Amit", "Rahul", "Deepak"]
        last_names = ["Sharma", "Verma", "Singh", "Patel", "Mehta", "Joshi", "Gupta", "Yadav", "Nair", "Rao"]
        
        first_name = first_names[hash_val % len(first_names)]
        last_name = last_names[(hash_val // len(first_names)) % len(last_names)]
        owner_name = f"{first_name} {last_name}"
        
        # Deterministic 10-digit number
        phone_suffix = str((hash_val * 12345) % 9000000000 + 1000000000)
        owner_phone = f"+91 {phone_suffix[:5]} {phone_suffix[5:]}"

        return {
            "owner_name": owner_name,
            "owner_phone": owner_phone
        }

    def compute_total_fine(self, violations: List[str]) -> int:
        """
        Computes the consolidated fine amount for a list of violations.

        Args:
            violations (List[str]): List of violation types.

        Returns:
            int: The total calculated fine in INR.
        """
        total = 0
        seen_violations = set()
        for v in violations:
            v_clean = v.upper().strip()
            # Prevent double-counting duplicates
            if v_clean in seen_violations:
                continue
            seen_violations.add(v_clean)
            
            # Retrieve fine, fallback to a standard ₹1000 if type is unknown
            fine = self.fine_mapping.get(v_clean, 1000)
            total += fine
            
        return total

    def _draw_annotations(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        violations: List[str],
        timestamp_str: str,
        plate_number: Optional[str]
    ) -> np.ndarray:
        """
        Draws bounding box annotations, violation types, and timestamp overlay on the frame.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # Bounding box coordinates
        x1, y1, x2, y2 = bbox
        # Clip to image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # 1. Bounding box around vehicle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # 2. Text label box above vehicle
        label_text = f"VIOLATION: {', '.join(violations)}"
        if plate_number:
            label_text += f" | PLATE: {plate_number}"
            
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Get text width & height
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        # Draw semi-transparent rectangle for background
        bg_x1 = x1
        bg_y1 = max(text_h + 15, y1 - 5)
        bg_x2 = min(w, x1 + text_w + 10)
        bg_y2 = y1 - text_h - 15 if y1 > text_h + 15 else y1 + text_h + 15
        
        # Solid red background label bar
        cv2.rectangle(annotated, (bg_x1, min(bg_y1, bg_y2)), (bg_x2, max(bg_y1, bg_y2)), (0, 0, 255), cv2.FILLED)
        
        # Draw white text in background bar
        text_y = y1 - 10 if y1 > text_h + 15 else y1 + text_h + 5
        cv2.putText(annotated, label_text, (x1 + 5, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # 3. Timestamp overlay banner (Top Left Corner)
        banner_text = f"AI TRAFFIC DETECTED | {timestamp_str}"
        (b_w, b_h), _ = cv2.getTextSize(banner_text, font, 0.5, 1)
        cv2.rectangle(annotated, (10, 10), (20 + b_w, 20 + b_h + 10), (0, 0, 0), cv2.FILLED)
        cv2.putText(annotated, banner_text, (15, 20 + b_h), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        return annotated

    def generate_challan(
        self,
        track_id: int,
        violations: List[str],
        bbox: Tuple[int, int, int, int],
        frame: np.ndarray,
        plate_number: Optional[str] = None,
        timestamp_epoch: Optional[float] = None,
        clip_frames: Optional[List[np.ndarray]] = None
    ) -> ChallanPayload:
        """
        Generates a complete challan: saves evidence, compiles a PDF document, and builds a payload dict.

        Args:
            track_id (int): Vehicle tracking identifier.
            violations (List[str]): List of violation types.
            bbox (Tuple[int, int, int, int]): Bounding box (x1, y1, x2, y2).
            frame (np.ndarray): Video frame matrix corresponding to violation moment.
            plate_number (str, optional): License plate text.
            timestamp_epoch (float, optional): Epoch timestamp. Defaults to current time.
            clip_frames (List[np.ndarray], optional): Circular buffer frames for violation clip extraction.

        Returns:
            ChallanPayload: Complete payload data.
        """
        # 1. Standardize inputs
        clean_violations = [v.upper().strip() for v in violations if v]
        if not clean_violations:
            clean_violations = ["GENERAL TRAFFIC VIOLATION"]

        # Date & Time formats
        dt_obj = datetime.fromtimestamp(timestamp_epoch) if timestamp_epoch else datetime.now()
        timestamp_iso = dt_obj.strftime("%Y-%m-%dT%H:%M:%S")
        timestamp_formatted = dt_obj.strftime("%d-%b-%Y %I:%M:%S %p")

        # 2. Get mock VAHAN lookup info
        owner_info = self.get_vehicle_owner_info(plate_number)
        owner_name = owner_info["owner_name"]
        owner_phone = owner_info["owner_phone"]
        license_plate = plate_number.upper().strip() if plate_number else "UNKNOWN"

        # 3. Unique Challan ID
        date_prefix = dt_obj.strftime("%Y%m%d")
        unique_suffix = str(uuid.uuid4().hex[:6]).upper()
        challan_id = f"CH-{date_prefix}-{unique_suffix}"

        # 4. Save evidence snapshot image
        annotated_frame = self._draw_annotations(frame, bbox, clean_violations, timestamp_formatted, plate_number)
        evidence_filename = f"evidence_{challan_id}.jpg"
        evidence_path = self.assets_dir / evidence_filename
        cv2.imwrite(str(evidence_path), annotated_frame)
        logger.info(f"Saved annotated violation evidence to: {evidence_path}")

        # 5. Save video clip if frames are provided
        clip_path_str = None
        if clip_frames:
            clip_filename = f"clip_{challan_id}.mp4"
            clip_path = self.outputs_dir / clip_filename
            self._save_clip_video(clip_frames, clip_path)
            clip_path_str = str(clip_path)

        # 6. Compute consolidated fine amount
        fine_amount = self.compute_total_fine(clean_violations)

        # 7. Generate PDF via ReportLab
        pdf_filename = f"CHALLAN_{challan_id}.pdf"
        pdf_path = self.challans_dir / pdf_filename
        self._compile_pdf(
            pdf_path=pdf_path,
            challan_id=challan_id,
            timestamp_str=timestamp_formatted,
            owner_name=owner_name,
            owner_phone=owner_phone,
            license_plate=license_plate,
            violations=clean_violations,
            fine_amount=fine_amount,
            evidence_img_path=evidence_path
        )
        logger.info(f"Successfully compiled PDF E-Challan: {pdf_path}")

        # 8. Build and return payload
        return ChallanPayload(
            challan_id=challan_id,
            timestamp=timestamp_iso,
            owner_name=owner_name,
            owner_phone=owner_phone,
            license_plate=license_plate,
            violation_type=", ".join(clean_violations),
            fine_amount=fine_amount,
            evidence_image_path=str(evidence_path),
            pdf_path=str(pdf_path),
            clip_path=clip_path_str
        )

    def _save_clip_video(self, frames: List[np.ndarray], output_path: Path):
        """Slices and saves frames to an MP4 video clip."""
        if not frames:
            return
        try:
            h, w = frames[0].shape[:2]
            try:
                # Try AVC1 (H.264) codec first
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out = cv2.VideoWriter(str(output_path), fourcc, 20.0, (w, h))
                if not out.isOpened():
                    raise Exception("avc1 writer not opened")
            except Exception:
                # Fallback to standard mp4v
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, 20.0, (w, h))
            
            for f in frames:
                out.write(f)
            out.release()
            logger.info(f"Saved video clip to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to write video clip: {e}")

    def _compile_pdf(
        self,
        pdf_path: Path,
        challan_id: str,
        timestamp_str: str,
        owner_name: str,
        owner_phone: str,
        license_plate: str,
        violations: List[str],
        fine_amount: int,
        evidence_img_path: Path
    ):
        """Compiles a professional ReportLab PDF E-Challan document."""
        # Simple doc template with standard page setup
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        
        # Premium color palette definitions
        primary_color = colors.HexColor("#1A365D")  # Navy Blue
        accent_color = colors.HexColor("#E53E3E")   # Crimson Red
        dark_text = colors.HexColor("#2D3748")      # Charcoal Grey
        light_bg = colors.HexColor("#F7FAFC")       # Off-white

        # Custom Paragraph styles
        title_style = ParagraphStyle(
            'ChallanTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=22,
            textColor=primary_color,
            alignment=1, # Centered
            spaceAfter=15
        )
        
        header_text_style = ParagraphStyle(
            'ChallanSub',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            textColor=colors.white,
            alignment=1
        )

        label_style = ParagraphStyle(
            'ChallanLabel',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=11,
            textColor=primary_color,
            spaceBefore=3,
            spaceAfter=3
        )

        value_style = ParagraphStyle(
            'ChallanVal',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=11,
            textColor=dark_text,
            spaceBefore=3,
            spaceAfter=3
        )

        fine_value_style = ParagraphStyle(
            'ChallanFineVal',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=12,
            textColor=accent_color,
            spaceBefore=3,
            spaceAfter=3
        )

        section_title_style = ParagraphStyle(
            'ChallanSec',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=14,
            textColor=primary_color,
            spaceBefore=15,
            spaceAfter=8
        )

        footer_style = ParagraphStyle(
            'ChallanFoot',
            parent=styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=9,
            textColor=colors.gray,
            alignment=1,
            spaceBefore=20
        )

        story = []

        # 1. Header banner Table
        header_data = [
            [Paragraph("<b>GOVERNMENT OF INDIA - MINISTRY OF ROAD TRANSPORT & HIGHWAYS</b>", header_text_style)],
            [Paragraph("<b>AUTOMATED TRAFFIC VIOLATION SYSTEM (E-CHALLAN)</b>", header_text_style)]
        ]
        header_table = Table(header_data, colWidths=[DocWidth := doc.width])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), primary_color),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 15))

        # 2. Main Title
        story.append(Paragraph("TRAFFIC E-CHALLAN RECEIPT", title_style))
        story.append(Spacer(1, 10))

        # 3. Challan Information Table (the 6 required fields + metadata)
        v_list_str = ", ".join(violations)
        
        info_data = [
            [Paragraph("Challan Reference ID", label_style), Paragraph(challan_id, value_style)],
            [Paragraph("Violation Date & Time", label_style), Paragraph(timestamp_str, value_style)],
            [Paragraph("Vehicle License Plate", label_style), Paragraph(license_plate, value_style)],
            [Paragraph("Registered Owner Name", label_style), Paragraph(owner_name, value_style)],
            [Paragraph("Owner Contact Number", label_style), Paragraph(owner_phone, value_style)],
            [Paragraph("Type of Violation(s)", label_style), Paragraph(v_list_str, value_style)],
            [Paragraph("<b>Total Fine Penalty</b>", label_style), Paragraph(f"<b>INR {fine_amount:,.2f}</b>", fine_value_style)]
        ]

        info_table = Table(info_data, colWidths=[DocWidth * 0.35, DocWidth * 0.65])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), light_bg),
            ('BOX', (0, 0), (-1, -1), 1, primary_color),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 20))

        # 4. Evidence snapshot section
        story.append(Paragraph("VIOLATION SNAPSHOT EVIDENCE", section_title_style))
        
        # Fit image to page width (DocWidth) while maintaining 16:9 ratio
        img_w = DocWidth
        img_h = DocWidth * (9.0 / 16.0)
        
        evidence_img = Image(str(evidence_img_path), width=img_w, height=img_h)
        story.append(evidence_img)
        story.append(Spacer(1, 25))

        # 5. Footer & Instructions
        disclaimer_text = (
            "Disclaimer: This is a computer-generated automated e-challan based on AI Object Detection "
            "and ANPR processing. Bounding box highlights represent tracked vehicle locations. "
            "Please pay the specified fine within 15 days via the official transport website portal. "
            "For disputes, contact the Traffic Headquarters."
        )
        story.append(Paragraph(disclaimer_text, footer_style))

        # Build PDF
        doc.build(story)
