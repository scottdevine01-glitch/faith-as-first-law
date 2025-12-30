#!/usr/bin/env python3
"""
Helper script for data collection in the speech compression experiment.
Helps manage participant scheduling, data organization, and quality checks.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
import sys

class DataCollectionManager:
    """Manages data collection for the speech compression experiment."""
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # File paths
        self.participants_file = self.data_dir / 'participants.csv'
        self.schedule_file = self.data_dir / 'schedule.csv'
        self.transcripts_dir = self.data_dir / 'transcripts'
        self.transcripts_dir.mkdir(exist_ok=True)
        
        # Initialize files if they don't exist
        self._initialize_files()
    
    def _initialize_files(self):
        """Initialize data files with headers if they don't exist."""
        if not self.participants_file.exists():
            pd.DataFrame(columns=[
                'participant_id', 'age', 'gender', 'education', 
                'consent_date', 'condition_order', 'status', 'notes'
            ]).to_csv(self.participants_file, index=False)
        
        if not self.schedule_file.exists():
            pd.DataFrame(columns=[
                'session_id', 'participant_id', 'session_date', 'condition',
                'audio_file', 'transcript_file', 'completed', 'quality_check'
            ]).to_csv(self.schedule_file, index=False)
    
    def register_participant(self, age, gender, education):
        """Register a new participant."""
        # Load existing participants
        participants = pd.read_csv(self.participants_file)
        
        # Generate participant ID
        if len(participants) == 0:
            new_id = 1
        else:
            # Extract numeric part from existing IDs
            existing_ids = participants['participant_id'].str.extract(r'P(\d+)').dropna()[0].astype(int)
            new_id = existing_ids.max() + 1
        
        participant_id = f"P{new_id:03d}"
        
        # Determine condition order (counterbalanced)
        orders = ['virtue,vice,neutral', 'vice,neutral,virtue', 'neutral,virtue,vice']
        order_index = new_id % 3
        condition_order = orders[order_index]
        
        # Add new participant
        new_participant = pd.DataFrame([{
            'participant_id': participant_id,
            'age': age,
            'gender': gender,
            'education': education,
            'consent_date': datetime.now().strftime('%Y-%m-%d'),
            'condition_order': condition_order,
            'status': 'registered',
            'notes': ''
        }])
        
        participants = pd.concat([participants, new_participant], ignore_index=True)
        participants.to_csv(self.participants_file, index=False)
        
        print(f"✓ Registered participant {participant_id}")
        print(f"  Condition order: {condition_order}")
        
        return participant_id
    
    def schedule_session(self, participant_id, session_date, condition):
        """Schedule a recording session."""
        schedule = pd.read_csv(self.schedule_file)
        
        # Check if participant exists
        participants = pd.read_csv(self.participants_file)
        if participant_id not in participants['participant_id'].values:
            print(f"✗ Participant {participant_id} not found")
            return False
        
        # Check if this condition was already completed
        existing = schedule[(schedule['participant_id'] == participant_id) & 
                           (schedule['condition'] == condition)]
        if len(existing) > 0:
            print(f"✗ Condition {condition} already completed for {participant_id}")
            return False
        
        # Generate session ID
        session_id = f"{participant_id}_{condition}_{session_date.replace('-', '')}"
        
        # Generate file names
        audio_file = f"{participant_id}_{condition}.wav"
        transcript_file = f"{participant_id}_{condition}.txt"
        
        # Add to schedule
        new_session = pd.DataFrame([{
            'session_id': session_id,
            'participant_id': participant_id,
            'session_date': session_date,
            'condition': condition,
            'audio_file': audio_file,
            'transcript_file': transcript_file,
            'completed': False,
            'quality_check': False
        }])
        
        schedule = pd.concat([schedule, new_session], ignore_index=True)
        schedule.to_csv(self.schedule_file, index=False)
        
        print(f"✓ Scheduled session {session_id}")
        print(f"  Files to create: {audio_file}, {transcript_file}")
        
        return True
    
    def complete_session(self, session_id, transcript_text, audio_duration, word_count):
        """Mark a session as completed with transcript."""
        schedule = pd.read_csv(self.schedule_file)
        
        if session_id not in schedule['session_id'].values:
            print(f"✗ Session {session_id} not found")
            return False
        
        # Update schedule
        schedule.loc[schedule['session_id'] == session_id, 'completed'] = True
        
        # Save transcript
        session = schedule[schedule['session_id'] == session_id].iloc[0]
        transcript_path = self.transcripts_dir / session['transcript_file']
        
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript_text)
        
        # Update schedule with metadata
        schedule.loc[schedule['session_id'] == session_id, 'audio_duration'] = audio_duration
        schedule.loc[schedule['session_id'] == session_id, 'word_count'] = word_count
        schedule.loc[schedule['session_id'] == session_id, 'completion_date'] = datetime.now().strftime('%Y-%m-%d')
        
        schedule.to_csv(self.schedule_file, index=False)
        
        print(f"✓ Completed session {session_id}")
        print(f"  Transcript saved: {transcript_path}")
        print(f"  Duration: {audio_duration}s, Words: {word_count}")
        
        return True
    
    def get_participant_progress(self, participant_id):
        """Get progress for a specific participant."""
        schedule = pd.read_csv(self.schedule_file)
        participant_sessions = schedule[schedule['participant_id'] == participant_id]
        
        if len(participant_sessions) == 0:
            print(f"No sessions found for {participant_id}")
            return
        
        completed = participant_sessions[participant_sessions['completed']]
        pending = participant_sessions[~participant_sessions['completed']]
        
        print(f"\nProgress for {participant_id}:")
        print(f"Completed: {len(completed)}/{len(participant_sessions)} sessions")
        
        if len(completed) > 0:
            print("\nCompleted sessions:")
            for _, session in completed.iterrows():
                print(f"  {session['condition']}: {session['session_date']}")
        
        if len(pending) > 0:
            print("\nPending sessions:")
            for _, session in pending.iterrows():
                print(f"  {session['condition']}: {session['session_date']}")
    
    def generate_data_summary(self):
        """Generate summary of collected data."""
        participants = pd.read_csv(self.participants_file)
        schedule = pd.read_csv(self.schedule_file)
        
        print("\n" + "="*60)
        print("DATA COLLECTION SUMMARY")
        print("="*60)
        
        print(f"\nParticipants: {len(participants)}")
        print(f"  Registered: {len(participants[participants['status'] == 'registered'])}")
        print(f"  Completed: {len(participants[participants['status'] == 'completed'])}")
        
        print(f"\nSessions: {len(schedule)}")
        print(f"  Completed: {len(schedule[schedule['completed']])}")
        print(f"  Pending: {len(schedule[~schedule['completed']])}")
        
        if 'condition' in schedule.columns:
            print(f"\nBy condition:")
            for condition in ['virtue', 'neutral', 'vice']:
                condition_sessions = schedule[schedule['condition'] == condition]
                completed = condition_sessions[condition_sessions['completed']]
                print(f"  {condition}: {len(completed)}/{len(condition_sessions)} completed")
        
        # Check for minimum sample size
        completed_by_condition = schedule[schedule['completed']].groupby('condition').size()
        min_completed = completed_by_condition.min() if len(completed_by_condition) > 0 else 0
        
        print(f"\nMinimum completed per condition: {min_completed}")
        
        if min_completed >= 20:
            print("✓ Sufficient sample size reached!")
        else:
            print(f"✗ Need {20 - min_completed} more per condition")

def main():
    parser = argparse.ArgumentParser(description='Manage speech compression data collection')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Register participant
    register_parser = subparsers.add_parser('register', help='Register a new participant')
    register_parser.add_argument('--age', type=int, required=True, help='Participant age')
    register_parser.add_argument('--gender', choices=['M', 'F', 'O'], required=True, help='Participant gender')
    register_parser.add_argument('--education', choices=['HS', 'BA', 'MA', 'PHD'], required=True, help='Education level')
    
    # Schedule session
    schedule_parser = subparsers.add_parser('schedule', help='Schedule a recording session')
    schedule_parser.add_argument('--participant', type=str, required=True, help='Participant ID')
    schedule_parser.add_argument('--date', type=str, required=True, help='Session date (YYYY-MM-DD)')
    schedule_parser.add_argument('--condition', choices=['virtue', 'neutral', 'vice'], required=True, help='Condition')
    
    # Complete session
    complete_parser = subparsers.add_parser('complete', help='Complete a session')
    complete_parser.add_argument('--session', type=str, required=True, help='Session ID')
    complete_parser.add_argument('--transcript', type=str, required=True, help='Transcript text file')
    complete_parser.add_argument('--duration', type=int, required=True, help='Audio duration in seconds')
    
    # Check progress
    progress_parser = subparsers.add_parser('progress', help='Check participant progress')
    progress_parser.add_argument('--participant', type=str, required=True, help='Participant ID')
    
    # Summary
    subparsers.add_parser('summary', help='Show data collection summary')
    
    args = parser.parse_args()
    
    manager = DataCollectionManager()
    
    if args.command == 'register':
        manager.register_participant(args.age, args.gender, args.education)
    
    elif args.command == 'schedule':
        manager.schedule_session(args.participant, args.date, args.condition)
    
    elif args.command == 'complete':
        # Read transcript from file
        with open(args.transcript, 'r', encoding='utf-8') as f:
            transcript_text = f.read()
        word_count = len(transcript_text.split())
        manager.complete_session(args.session, transcript_text, args.duration, word_count)
    
    elif args.command == 'progress':
        manager.get_participant_progress(args.participant)
    
    elif args.command == 'summary':
        manager.generate_data_summary()
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
