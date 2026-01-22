import streamlit as st
import pandas as pd
from datetime import datetime
import re

class FinanceQAManager:
    def __init__(self):
        # Initialize Q&A storage in session state
        if 'qa_database' not in st.session_state:
            st.session_state.qa_database = []
            # Add some sample Q&As
            self._add_sample_data()
    
    def _add_sample_data(self):
        """Add sample Q&A data"""
        sample_qas = [
            {
                'id': 1,
                'question': 'What is the best way to start investing in stocks?',
                'answer': 'Start with index funds or ETFs for diversification. Research companies before individual stock picks. Only invest money you can afford to lose.',
                'category': 'investment',
                'author': 'FinanceExpert',
                'timestamp': datetime.now(),
                'votes': 5,
                'tags': ['stocks', 'investing', 'beginner']
            },
            {
                'id': 2,
                'question': 'How much should I save for emergency fund?',
                'answer': 'Aim for 3-6 months of living expenses. Keep it in a high-yield savings account for easy access.',
                'category': 'savings',
                'author': 'SaverPro',
                'timestamp': datetime.now(),
                'votes': 8,
                'tags': ['emergency fund', 'savings', 'financial planning']
            },
            {
                'id': 3,
                'question': 'What tax deductions can I claim as a freelancer?',
                'answer': 'Home office expenses, business equipment, professional development, travel expenses, and software subscriptions are common deductions.',
                'category': 'tax',
                'author': 'TaxAdvisor',
                'timestamp': datetime.now(),
                'votes': 12,
                'tags': ['tax', 'freelancer', 'deductions', 'self-employed']
            }
        ]
        st.session_state.qa_database.extend(sample_qas)
    
    def add_question(self, question, category, author="Anonymous", tags=None):
        """Add a new question to the database"""
        new_id = max([qa['id'] for qa in st.session_state.qa_database], default=0) + 1
        
        new_question = {
            'id': new_id,
            'question': question,
            'answer': None,
            'category': category,
            'author': author,
            'timestamp': datetime.now(),
            'votes': 0,
            'tags': tags or []
        }
        
        st.session_state.qa_database.append(new_question)
        return new_id
    
    def add_answer(self, question_id, answer, author="Anonymous"):
        """Add an answer to an existing question"""
        for qa in st.session_state.qa_database:
            if qa['id'] == question_id:
                qa['answer'] = answer
                qa['answer_author'] = author
                qa['answer_timestamp'] = datetime.now()
                return True
        return False
    
    def search_questions(self, keyword="", category="all"):
        """Search questions based on keyword and category"""
        results = []
        
        for qa in st.session_state.qa_database:
            # Category filter
            if category != "all" and qa['category'] != category:
                continue
            
            # Keyword search in question, answer, and tags
            if keyword:
                keyword_lower = keyword.lower()
                search_text = f"{qa['question']} {qa.get('answer', '')} {' '.join(qa['tags'])}".lower()
                if keyword_lower not in search_text:
                    continue
            
            results.append(qa)
        
        # Sort by votes (descending) and then by timestamp (newest first)
        results.sort(key=lambda x: (-x['votes'], -x['timestamp'].timestamp()))
        return results
    
    def upvote_question(self, question_id):
        """Upvote a question"""
        for qa in st.session_state.qa_database:
            if qa['id'] == question_id:
                qa['votes'] += 1
                return True
        return False
    
    def get_question_by_id(self, question_id):
        """Get a specific question by ID"""
        for qa in st.session_state.qa_database:
            if qa['id'] == question_id:
                return qa
        return None
    
    def get_categories(self):
        """Get all unique categories"""
        categories = set()
        for qa in st.session_state.qa_database:
            categories.add(qa['category'])
        return sorted(list(categories))
    
    def get_popular_tags(self, limit=10):
        """Get most popular tags"""
        tag_count = {}
        for qa in st.session_state.qa_database:
            for tag in qa['tags']:
                tag_count[tag] = tag_count.get(tag, 0) + 1
        
        # Sort by count and return top tags
        sorted_tags = sorted(tag_count.items(), key=lambda x: x[1], reverse=True)
        return [tag for tag, count in sorted_tags[:limit]]