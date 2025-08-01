from imports import * 

class FlightSearchArgs(BaseModel):
    departure: str = Field(description="The departure city or airport code (e.g., 'Mumbai', 'BOM').")
    destination: str = Field(description="The destination city or airport code (e.g., 'Paris', 'CDG').")
    departure_date: str = Field(description="The departure date in YYYY-MM-DD format.")
    return_date: Optional[str] = Field(None, description="The return date in YYYY-MM-DD format (optional).")
    passengers: int = Field(1, description="The number of passengers.")

class HotelSearchArgs(BaseModel):
    location: str = Field(description="The city or location to search for hotels.")
    check_in: str = Field(description="The check-in date in YYYY-MM-DD format.")
    check_out: str = Field(description="The check-out date in YYYY-MM-DD format.")
    guests: int = Field(2, description="The number of guests.")
    preferences: Optional[str] = Field(None, description="User preferences for the hotel, e.g., 'luxury', 'budget', '5-star'.")

class EmailArgs(BaseModel):
    to_email: str = Field(description="The recipient's email address.")
    subject: str = Field(description="The subject line of the email.")
    content: str = Field(description="The HTML or plain text content of the email body.")

class CompleteTripArgs(BaseModel):
    departure: str = Field(description="The departure city or airport code for the trip.")
    destination: str = Field(description="The destination city for the trip.")
    departure_date: str = Field(description="The departure date for the trip in YYYY-MM-DD format.")
    return_date: str = Field(description="The return date for the trip in YYYY-MM-DD format.")
    hotel_preferences: Optional[str] = Field(None, description="User preferences for the hotel, e.g., 'luxury', 'budget'.")
    email_address: Optional[str] = Field(None, description="Optional email address to send the final itinerary to.")