# streamlit_app.py
import streamlit as st
import json
import time
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import pandas as pd

# Your existing agent code (simplified for Streamlit)
class TaskType(Enum):
    ANALYSIS = "analysis"
    SEARCH = "search"
    CALCULATION = "calculation"
    GENERATION = "generation"
    DECISION = "decision"

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    id: str
    type: TaskType
    description: str
    input_data: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class StreamlitAIAgent:
    def __init__(self, name: str = "WebAgent"):
        self.name = name
        self.capabilities = ["text analysis", "calculations", "knowledge search", "response generation"]
        
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for basic metrics"""
        words = text.split()
        sentences = text.split('.')
        return {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "avg_word_length": round(sum(len(w) for w in words) / len(words) if words else 0, 2),
            "sentiment": "neutral"
        }
    
    def calculate(self, expression: str) -> float:
        """Safe calculator for basic math"""
        try:
            allowed_chars = set('0123456789+-*/.() ')
            if all(c in allowed_chars for c in expression):
                return eval(expression)
            else:
                raise ValueError("Invalid characters in expression")
        except Exception as e:
            raise ValueError(f"Calculation error: {e}")
    
    def search_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """Simulate knowledge search"""
        topics = {
            "weather": {"result": "Currently sunny, 22°C", "confidence": 0.9},
            "time": {"result": f"Current time: {datetime.now().strftime('%H:%M:%S')}", "confidence": 1.0},
            "python": {"result": "Python is a programming language", "confidence": 0.8},
            "ai": {"result": "AI stands for Artificial Intelligence", "confidence": 0.9},
            "streamlit": {"result": "Streamlit is a Python framework for building web apps", "confidence": 0.9}
        }
        
        results = []
        for topic, data in topics.items():
            if topic.lower() in query.lower():
                results.append({
                    "topic": topic,
                    "content": data["result"],
                    "confidence": data["confidence"]
                })
        
        return results if results else [{"topic": "general", "content": "No specific information found", "confidence": 0.1}]
    
    def generate_response(self, prompt: str) -> str:
        """Generate a response based on prompt"""
        if "hello" in prompt.lower():
            return f"Hello! I'm {self.name}, your AI assistant deployed on Streamlit! 🚀"
        elif "how are you" in prompt.lower():
            return "I'm running smoothly on Streamlit Community Cloud! 💪"
        elif "what can you do" in prompt.lower():
            return f"I can help with: {', '.join(self.capabilities)}"
        else:
            return f"I understand you're asking about: {prompt}. Let me help you with that."
    
    def make_decision(self, options: List[str], criteria: Dict = None) -> Dict[str, Any]:
        """Make a decision from given options"""
        if not options:
            return {"decision": None, "reason": "No options provided"}
        
        scored_options = []
        for option in options:
            score = random.uniform(0.3, 1.0)
            scored_options.append({"option": option, "score": round(score, 2)})
        
        best_option = max(scored_options, key=lambda x: x["score"])
        
        return {
            "decision": best_option["option"],
            "score": best_option["score"],
            "reason": "Selected based on scoring criteria",
            "all_scores": scored_options
        }

# Initialize agent
@st.cache_resource
def get_agent():
    return StreamlitAIAgent("StreamlitBot")

# Streamlit App
def main():
    st.set_page_config(
        page_title="AI Agent Dashboard",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize agent
    agent = get_agent()
    
    # App header
    st.title("🤖 AI Agent Dashboard")
    st.markdown("**Your AI Assistant deployed on Streamlit Community Cloud**")
    
    # Sidebar for agent info
    with st.sidebar:
        st.header("🔧 Agent Info")
        st.write(f"**Name:** {agent.name}")
        st.write(f"**Status:** 🟢 Active")
        st.write("**Capabilities:**")
        for cap in agent.capabilities:
            st.write(f"• {cap}")
        
        st.header("📊 Quick Stats")
        if 'task_count' not in st.session_state:
            st.session_state.task_count = 0
        st.metric("Tasks Completed", st.session_state.task_count)
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat", "📝 Text Analysis", "🔢 Calculator", "🔍 Search", "🎯 Decision Maker"])
    
    # Chat Tab
    with tab1:
        st.header("💬 Chat with Agent")
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Chat input
        user_input = st.text_input("Type your message:", key="chat_input")
        
        if st.button("Send", key="send_chat"):
            if user_input:
                # Add user message
                st.session_state.chat_history.append({"role": "user", "message": user_input})
                
                # Get agent response
                response = agent.generate_response(user_input)
                st.session_state.chat_history.append({"role": "agent", "message": response})
                
                st.session_state.task_count += 1
                st.rerun()
        
        # Display chat history
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.chat_message("user").write(chat["message"])
            else:
                st.chat_message("assistant").write(chat["message"])
    
    # Text Analysis Tab
    with tab2:
        st.header("📝 Text Analysis")
        
        text_input = st.text_area("Enter text to analyze:", height=150)
        
        if st.button("Analyze Text", key="analyze_btn"):
            if text_input:
                with st.spinner("Analyzing..."):
                    result = agent.analyze_text(text_input)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Word Count", result["word_count"])
                    with col2:
                        st.metric("Sentences", result["sentence_count"])
                    with col3:
                        st.metric("Avg Word Length", result["avg_word_length"])
                    
                    st.success(f"Sentiment: {result['sentiment']}")
                    st.session_state.task_count += 1
    
    # Calculator Tab
    with tab3:
        st.header("🔢 Calculator")
        
        calc_input = st.text_input("Enter mathematical expression:", placeholder="e.g., 2 + 3 * 4")
        
        if st.button("Calculate", key="calc_btn"):
            if calc_input:
                try:
                    with st.spinner("Calculating..."):
                        result = agent.calculate(calc_input)
                        st.success(f"Result: {result}")
                        st.session_state.task_count += 1
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Search Tab
    with tab4:
        st.header("🔍 Knowledge Search")
        
        search_query = st.text_input("Enter search query:", placeholder="e.g., python, ai, weather")
        
        if st.button("Search", key="search_btn"):
            if search_query:
                with st.spinner("Searching..."):
                    results = agent.search_knowledge(search_query)
                    
                    for result in results:
                        with st.expander(f"📋 {result['topic'].title()}"):
                            st.write(result['content'])
                            st.progress(result['confidence'])
                            st.caption(f"Confidence: {result['confidence']:.1%}")
                    
                    st.session_state.task_count += 1
    
    # Decision Maker Tab
    with tab5:
        st.header("🎯 Decision Maker")
        
        st.write("Enter options (one per line):")
        options_text = st.text_area("Options:", height=100, placeholder="Option A\nOption B\nOption C")
        
        if st.button("Make Decision", key="decision_btn"):
            if options_text:
                options = [opt.strip() for opt in options_text.split('\n') if opt.strip()]
                
                with st.spinner("Making decision..."):
                    decision = agent.make_decision(options)
                    
                    if decision["decision"]:
                        st.success(f"🎯 **Decision:** {decision['decision']}")
                        st.info(f"📊 **Score:** {decision['score']}")
                        st.write(f"**Reason:** {decision['reason']}")
                        
                        # Show all scores
                        st.subheader("📈 All Scores")
                        scores_df = pd.DataFrame(decision['all_scores'])
                        st.dataframe(scores_df, use_container_width=True)
                        
                        st.session_state.task_count += 1
                    else:
                        st.error("No valid options provided")
    
    # Footer
    st.markdown("---")
    st.markdown("🚀 **Deployed on Streamlit Community Cloud** | Built with ❤️ using Python")

if __name__ == "__main__":
    main()
