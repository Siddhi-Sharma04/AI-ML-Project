import logging
from typing import Dict

logger = logging.getLogger("VahanLookupService")

class VahanLookupService:
    def __init__(self):
        # Local mock registry
        self.vahan_mock_db = {
            "RJ14AB1234": {"owner_name": "Niyati Kumawat", "owner_phone": "+91 98765 43210"},
            "MH12CD5678": {"owner_name": "Ramesh Kumar", "owner_phone": "+91 99999 88888"},
            "DL3CA5555": {"owner_name": "Priya Sharma", "owner_phone": "+91 98111 22222"},
            "KA03MM9999": {"owner_name": "Anil Kumble", "owner_phone": "+91 94444 55555"}
        }

    async def get_vehicle_owner_info(self, plate_number: str) -> Dict[str, str]:
        """
        Retrieves vehicle owner information deterministically.
        If plate is unknown, generates details based on characters.
        """
        if not plate_number:
            return {
                "owner_name": "Unknown Vehicle Owner",
                "owner_phone": "+91 00000 00000"
            }

        # Clean plate characters
        clean_plate = "".join(e for e in plate_number if e.isalnum()).upper().strip()

        if clean_plate in self.vahan_mock_db:
            logger.info(f"VAHAN DB Match found for plate: {clean_plate}")
            return self.vahan_mock_db[clean_plate]

        # Deterministic generation for unknown plates
        hash_val = sum(ord(char) for char in clean_plate)
        
        first_names = ["Rajesh", "Vikram", "Sanjay", "Karan", "Sunita", "Anjali", "Neha", "Amit", "Rahul", "Deepak"]
        last_names = ["Sharma", "Verma", "Singh", "Patel", "Mehta", "Joshi", "Gupta", "Yadav", "Nair", "Rao"]
        
        first_name = first_names[hash_val % len(first_names)]
        last_name = last_names[(hash_val // len(first_names)) % len(last_names)]
        owner_name = f"{first_name} {last_name}"
        
        phone_suffix = str((hash_val * 12345) % 9000000000 + 1000000000)
        owner_phone = f"+91 {phone_suffix[:5]} {phone_suffix[5:]}"

        logger.info(f"VAHAN DB Fallback mock generated for plate: {clean_plate}")
        return {
            "owner_name": owner_name,
            "owner_phone": owner_phone
        }

vahan_lookup_service = VahanLookupService()
