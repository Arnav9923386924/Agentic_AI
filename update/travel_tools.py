import json
from datetime import datetime
from typing import Optional, Any, Callable # Added Callable and Any for type hints

from imports import *
from manager import AgentMemory , AgentState , Task
from tools.learning_tools import *
from tools.web_tools import *
from tools.system_tools import *
from tools.planning_tools import *
from tools.learning_tools import *


console = Console() 


class TravelTools:
    """
    A dedicated class that groups all travel-related tools and logic.
    These methods are designed to be called by the agent executor.
    """
    def __init__(self, memory: AgentMemory, llm: Any, get_user_context_func: Callable[[], dict],
                 serpapi_key: str, sendgrid_api_key: str, sendgrid_from_email: str):
        """
        Initializes the TravelTools class with dependencies from the main agent and API keys.
        
        Args:
            memory (AgentMemory): The agent's memory manager.
            llm (Any): The large language model instance.
            get_user_context_func (Callable[[], dict]): A function to retrieve user context from the main agent.
            serpapi_key (str): Your SerpAPI API key.
            sendgrid_api_key (str): Your SendGrid API key.
            sendgrid_from_email (str): The email address to send emails from.
        """
        self.memory = memory
        self.llm = llm
        self.get_user_context = get_user_context_func # Store the function reference
        self.serpapi_key = serpapi_key          # Store the key
        self.sendgrid_api_key = sendgrid_api_key  # Store the key
        self.sendgrid_from_email = sendgrid_from_email # Store the email

    def search_flights(self, departure: str, destination: str, departure_date: str, return_date: str = None, passengers: int = 1) -> str:
        """Search for flights using SerpAPI Google Flights.
        
        Args:
            departure (str): The departure city or airport code.
            destination (str): The destination city or airport code.
            departure_date (str): The departure date in YYYY-MM-DD format.
            return_date (str, optional): The return date in YYYY-MM-DD format. Defaults to None (one-way).
            passengers (int, optional): Number of passengers. Defaults to 1.
        
        Returns:
            str: A formatted string of flight search results or an error message.
        """
        try:
            if not self.serpapi_key:
                return "❌ SerpAPI key not configured for TravelTools."
            
            console.print(f"[cyan]✈️ Searching flights from {departure} to {destination} for {departure_date}...[/cyan]")
            
            params = {
                "engine": "google_flights",
                "departure_id": departure,
                "arrival_id": destination, 
                "outbound_date": departure_date,
                "currency": "USD",
                "hl": "en",
                "api_key": self.serpapi_key
            }
            
            if return_date:
                params["return_date"] = return_date
                params["type"] = "1"  # Round trip
            else:
                params["type"] = "2"  # One way
                
            search = GoogleSearch(params)
            results = search.get_dict()
            
            if "best_flights" not in results:
                return "❌ No flights found for the specified route and date."
            
            flight_results_str = "✈️ Flight Search Results:\n\n"
            
            # Limit to top 5 best flights if available
            for i, flight in enumerate(results.get("best_flights", [])[:5], 1):
                # Each 'flight' in best_flights can have multiple segments/options.
                # Just taking the first flight_info for summary.
                if flight.get("flights"):
                    flight_info = flight["flights"][0] 
                    airline = flight_info.get("airline", "Unknown Airline")
                    departure_time = flight_info.get("departure_airport", {}).get("time", "N/A")
                    arrival_time = flight_info.get("arrival_airport", {}).get("time", "N/A")
                    duration = flight_info.get("duration", "N/A")
                    price = flight.get("price", "N/A")

                    flight_results_str += f"{i}. {airline}\n"
                    flight_results_str += f"  Departure: {departure_time} | Arrival: {arrival_time}\n"
                    flight_results_str += f"  Duration: {duration} | Price: {price}\n\n"
            
            # Store search in memory for future reference
            self.memory.update_user_profile("last_flight_search", {
                "departure": departure,
                "destination": destination,
                "date": departure_date,
                "results": results.get("best_flights", [])[:5], # Store only top 5 for brevity
                "timestamp": datetime.now().isoformat()
            })
            
            console.print(f"[green]{flight_results_str}[/green]")
            return flight_results_str
            
        except Exception as e:
            error_msg = f"❌ Flight search error: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            return error_msg

    def search_hotels(self, location: str, check_in: str, check_out: str, guests: int = 2, preferences: Optional[str] = None) -> str:
        """Search for hotels using SerpAPI Google Hotels.
        
        Args:
            location (str): The city or area for hotels.
            check_in (str): Check-in date in YYYY-MM-DD format.
            check_out (str): Check-out date in YYYY-MM-DD format.
            guests (int, optional): Number of guests. Defaults to 2.
            preferences (str, optional): Specific hotel preferences (e.g., "luxury", "budget").
        
        Returns:
            str: A formatted string of hotel search results or an error message.
        """
        try:
            if not self.serpapi_key:
                return "❌ SerpAPI key not configured for TravelTools."

            query = f"{preferences} hotels in {location}" if preferences else location
            console.print(f"[cyan]🏨 Searching for hotels in: '{query}' from {check_in} to {check_out}...[/cyan]")

            params = {
                "engine": "google_hotels",
                "q": query,
                "check_in_date": check_in,
                "check_out_date": check_out,
                "adults": guests,
                "currency": "USD",
                "gl": "us",
                "hl": "en",
                "api_key": self.serpapi_key
            }
            search = GoogleSearch(params)
            results = search.get_dict()

            if "properties" not in results:
                return f"❌ No hotels found for '{query}' with the specified dates."

            hotel_results_str = "🏨 Hotel Search Results:\n\n"
            for i, hotel in enumerate(results.get("properties", [])[:5], 1): # Limit to top 5 properties
                name = hotel.get("name", "Unknown Hotel")
                rate = hotel.get("rate_per_night", {}).get("lowest", "N/A")
                rating = hotel.get("overall_rating", "N/A")
                reviews = hotel.get("reviews", "N/A")
                hotel_results_str += f"{i}. {name}\n"
                hotel_results_str += f" Rating: {rating}⭐ ({reviews} reviews)\n"
                hotel_results_str += f" Price: ${rate}/night\n\n"

            # Store search in memory
            self.memory.update_user_profile("last_hotel_search", {
                "location": location,
                "check_in": check_in,
                "check_out": check_out,
                "results": results.get("properties", [])[:5], # Store only top 5
                "timestamp": datetime.now().isoformat()
            })

            console.print(f"[green]{hotel_results_str}[/green]")
            return hotel_results_str
        except Exception as e:
            error_msg = f"❌ Hotel search error: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            return error_msg

    def send_travel_email(self, to_email: str, subject: str, content: str, travel_type: str = "itinerary") -> str:
        """Send personalized travel emails using SendGrid.
        
        Args:
            to_email (str): Recipient's email address.
            subject (str): Subject of the email.
            content (str): The main body content of the email.
            travel_type (str, optional): Type of travel email (e.g., "itinerary", "confirmation"). Defaults to "itinerary".
        
        Returns:
            str: Success message if email sent, or an error message.
        """
        try:
            if not self.sendgrid_api_key:
                return "❌ SendGrid API key not configured for TravelTools."
            if not self.sendgrid_from_email:
                return "❌ SendGrid 'from' email not configured for TravelTools."
            
            console.print(f"[cyan]📧 Sending {travel_type} email to {to_email} with subject '{subject}'...[/cyan]")
            
            # Get user context for personalization using the passed-in function
            user_context = self.get_user_context()
            user_name = user_context.get('personal_info', {}).get('name', 'Traveler')
            
            # Create personalized email template
            html_content = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <header style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; border-radius: 10px 10px 0 0;">
                        <h1>✈️ Your Travel Assistant</h1>
                    </header>
                    
                    <main style="background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px;">
                        <h2>Hello {user_name}! 👋</h2>
                        <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                            {content.replace(chr(10), '<br>')}
                        </div>
                        
                        <footer style="margin-top: 30px; padding-top: 20px; border-top: 2px solid #ddd; text-align: center; color: #666;">
                            <p>🤖 Sent by AIA - Your Autonomous Intelligent Assistant</p>
                            <p style="font-size: 12px;">This email was generated based on your travel preferences and recent interactions.</p>
                        </footer>
                    </main>
                </div>
            </body>
            </html>
            """
            
            message = Mail(
                from_email=self.sendgrid_from_email,
                to_emails=to_email,
                subject=f"✈️ {subject}",
                html_content=html_content
            )
            
            sg = SendGridAPIClient(api_key=self.sendgrid_api_key)
            response = sg.send(message)
            
            # Store email in memory
            self.memory.update_user_profile("last_email_sent", {
                "to": to_email,
                "subject": subject,
                "type": travel_type,
                "timestamp": datetime.now().isoformat(),
                "status": "sent"
            })
            
            success_msg = f"📧 Email sent successfully to {to_email} (Status Code: {response.status_code})"
            console.print(f"[green]{success_msg}[/green]")
            return success_msg
            
        except Exception as e:
            error_msg = f"❌ Email sending error: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            return error_msg

    def plan_complete_trip(self, destination: str, departure_date: str, return_date: str, email_address: str = None, hotel_preferences: Optional[str] = None, departure: Optional[str] = None) -> str:
        """Plan a complete trip with flights, hotels, and email itinerary.
        
        Args:
            destination (str): The final destination city.
            departure_date (str): The departure date in YYYY-MM-DD format.
            return_date (str): The return date in YYYY-MM-DD format.
            email_address (str, optional): Email to send the itinerary to. Defaults to None.
            hotel_preferences (str, optional): Preferences for hotel search. Defaults to None.
            departure (str, optional): Departure city if different from user's usual location. Defaults to None.
        
        Returns:
            str: A formatted string of the complete trip plan or an error message.
        """
        try:
            console.print(f"[yellow]🌍 Planning complete trip to {destination} from {departure_date} to {return_date}...[/yellow]")
            
            # Get user's location from profile using self.get_user_context
            user_context = self.get_user_context()
            user_location = departure or user_context.get('personal_info', {}).get('location', 'Mumbai') # Default 'Mumbai'
            
            trip_plan = f"🎯 Complete Trip Plan to {destination}\n\n"
            
            # 1. Search flights by calling the method from THIS class (self.search_flights)
            flight_results = self.search_flights(
                departure=user_location,
                destination=destination, 
                departure_date=departure_date,
                return_date=return_date,
                passengers=1 # Assuming 1 passenger for trip planning
            )
            trip_plan += flight_results + "\n"
            
            # 2. Search hotels by calling the method from THIS class (self.search_hotels)
            hotel_results = self.search_hotels(
                location=destination,
                check_in=departure_date, # Using departure date as check-in for simplicity
                check_out=return_date,
                guests=2, # Assuming 2 guests for hotel search
                preferences=hotel_preferences
            )
            trip_plan += hotel_results + "\n"
            
            # 3. Generate travel tips using LLM (self.llm)
            if self.llm:
                tips_prompt = f"""Generate 5 personalized travel tips for someone visiting {destination} 
                from {departure_date} to {return_date}. Include local attractions, weather considerations, 
                and cultural insights. Make it practical and engaging."""
                
                tips_response = self.llm.invoke(tips_prompt)
                travel_tips = f"💡 Travel Tips for {destination}:\n{tips_response.content}\n\n"
                trip_plan += travel_tips
            else:
                trip_plan += "⚠️ LLM not available for generating travel tips.\n\n"

            # 4. Send email if requested by calling the method from THIS class (self.send_travel_email)
            if email_address:
                email_subject = f"Your Trip Itinerary to {destination}"
                email_sent_status = self.send_travel_email( # Use a variable name for status
                    to_email=email_address, 
                    subject=email_subject, 
                    content=trip_plan, 
                    travel_type="complete_itinerary"
                )
                trip_plan += f"\n{email_sent_status}" # Append the status message
            
            # Store complete trip plan using self.memory
            self.memory.update_user_profile("last_trip_plan", {
                "destination": destination,
                "dates": f"{departure_date} to {return_date}",
                "plan": trip_plan,
                "timestamp": datetime.now().isoformat()
            })
            
            console.print(f"[green]✅ Trip planning complete for {destination}.[/green]")
            return trip_plan       
        except Exception as e:
            error_msg = f"❌ Trip planning error: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            return error_msg

