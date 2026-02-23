import { useState, useEffect } from 'react';
import { meetingsApi } from '../services/api';
import type { Meeting } from '../types';

export const useMeetings = (userEmail?: string, status?: string, upcomingOnly?: boolean, asAdmin?: boolean) => {
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMeetings = async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await meetingsApi.list(userEmail, status, upcomingOnly, asAdmin);
        setMeetings(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Error al cargar reuniones');
      } finally {
        setLoading(false);
      }
    };

    fetchMeetings();
  }, [userEmail, status, upcomingOnly, asAdmin]);

  return { meetings, loading, error, refetch: () => {
    const fetchMeetings = async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await meetingsApi.list(userEmail, status, upcomingOnly, asAdmin);
        setMeetings(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Error al cargar reuniones');
      } finally {
        setLoading(false);
      }
    };
    fetchMeetings();
  } };
};

